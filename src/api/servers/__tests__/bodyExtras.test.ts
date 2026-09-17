import type {ReasoningIntent} from '../../../utils/completionTypes';
import {EFFORT_LEVELS} from '../../../utils/reasoningCapability';
import type {SamplerParam, Samplers} from '../../../utils/samplerParams';
import {slotsAfterSamplerRequest} from '../../../../jest/fixtures/llamaServerWire';
import {PROPS_READ_NAMES} from '../../llamaServer/props';
import {dialectFor} from '../index';
import {llamaCpp} from '../llamaCpp';

const reasoningBody = (
  serverType: string | undefined,
  reasoning?: ReasoningIntent,
) => dialectFor(serverType).bodyExtras({samplers: {}, reasoning});

const samplerBody = (serverType: string | undefined, samplers: Samplers) =>
  dialectFor(serverType).bodyExtras({samplers});

describe('bodyExtras (per-serverType reasoning gating)', () => {
  it('returns empty when no reasoning intent', () => {
    expect(reasoningBody('llama.cpp', undefined)).toEqual({});
  });

  it('llama.cpp ON sends reasoning_format auto', () => {
    expect(reasoningBody('llama.cpp', {enabled: true})).toEqual({
      reasoning_format: 'auto',
    });
  });

  it('llama.cpp ON+effort sends reasoning_format auto + reasoning_effort', () => {
    expect(reasoningBody('llama.cpp', {enabled: true, effort: 'high'})).toEqual(
      {
        reasoning_format: 'auto',
        chat_template_kwargs: {reasoning_effort: 'high'},
        reasoning_budget_tokens: 8192,
      },
    );
  });

  it('llama.cpp budgets rise with the effort level and uncap at max', () => {
    const budgetFor = (effort: string) =>
      reasoningBody('llama.cpp', {enabled: true, effort})
        .reasoning_budget_tokens;

    expect(EFFORT_LEVELS.map(budgetFor)).toEqual([
      256, 512, 2048, 8192, 16384, -1,
    ]);
  });

  it('llama.cpp sends an effort that is not a level, but no budget for it', () => {
    expect(
      reasoningBody('llama.cpp', {enabled: true, effort: 'extreme'}),
    ).toEqual({
      reasoning_format: 'auto',
      chat_template_kwargs: {reasoning_effort: 'extreme'},
    });
  });

  it('llama.cpp sends no budget when reasoning is on without an effort', () => {
    const on = reasoningBody('llama.cpp', {enabled: true});
    expect(on).not.toHaveProperty('reasoning_budget_tokens');
    const off = reasoningBody('llama.cpp', {enabled: false});
    expect(off).not.toHaveProperty('reasoning_budget_tokens');
  });

  it('sends no budget to any other server type', () => {
    for (const serverType of [
      'vLLM',
      'LM Studio',
      'Ollama',
      'OpenAI',
      undefined,
    ]) {
      expect(
        reasoningBody(serverType, {enabled: true, effort: 'high'}),
      ).not.toHaveProperty('reasoning_budget_tokens');
    }
  });

  it('llama.cpp OFF sends enable_thinking:false + reasoning_format auto', () => {
    const off = reasoningBody('llama.cpp', {enabled: false});
    expect(off).toEqual({
      reasoning_format: 'auto',
      chat_template_kwargs: {enable_thinking: false},
    });
    // llama-server has no top-level reasoning_effort: it 200s and keeps
    // thinking on, so enable_thinking is the only thing that turns it off.
    expect(off).not.toHaveProperty('reasoning_effort');
  });

  it('LM Studio is on/off only — no graded effort', () => {
    expect(reasoningBody('LM Studio', {enabled: false})).toEqual({
      chat_template_kwargs: {enable_thinking: false},
    });
    expect(reasoningBody('LM Studio', {enabled: true})).toEqual({});
    // Even when an effort is set, LM Studio never sends reasoning_effort.
    const onEffort = reasoningBody('LM Studio', {
      enabled: true,
      effort: 'high',
    });
    expect(onEffort).toEqual({});
    expect(onEffort).not.toHaveProperty('reasoning_effort');
  });

  it('vLLM ON+effort sends chat_template_kwargs.reasoning_effort', () => {
    expect(reasoningBody('vLLM', {enabled: true, effort: 'max'})).toEqual({
      chat_template_kwargs: {reasoning_effort: 'max'},
    });
    // ON without an effort sends nothing.
    expect(reasoningBody('vLLM', {enabled: true})).toEqual({});
  });

  it('vLLM OFF sends enable_thinking:false', () => {
    expect(reasoningBody('vLLM', {enabled: false})).toEqual({
      chat_template_kwargs: {enable_thinking: false},
    });
  });

  it('Ollama OFF sends only reasoning_effort none and never think:true', () => {
    const off = reasoningBody('Ollama', {enabled: false});
    expect(off).toEqual({reasoning_effort: 'none'});
    expect(off).not.toHaveProperty('think');
    // ON sends nothing — never think:true, never a non-none effort.
    const on = reasoningBody('Ollama', {enabled: true, effort: 'high'});
    expect(on).toEqual({});
    expect(on).not.toHaveProperty('think');
    expect(on).not.toHaveProperty('reasoning_effort');
  });

  it('OpenAI sends reasoning_effort only when effort is known', () => {
    expect(reasoningBody('OpenAI', {enabled: true, effort: 'medium'})).toEqual({
      reasoning_effort: 'medium',
    });
    // No effort known → omit everything (no enable_thinking, no 400 bait).
    expect(reasoningBody('OpenAI', {enabled: true})).toEqual({});
    expect(reasoningBody('OpenAI', {enabled: false})).toEqual({});
  });

  it('unknown serverType omits everything', () => {
    expect(reasoningBody(undefined, {enabled: false})).toEqual({});
    expect(
      reasoningBody('something-else', {enabled: false, effort: 'low'}),
    ).toEqual({});
  });
});

describe('bodyExtras (sampler forwarding)', () => {
  // The read-back of a request carrying the `settings` below, so the slot
  // holds both the server's own vocabulary and the values it accepted under
  // it. Projected from the live body rather than restated here: a name spelled
  // the same way in the parser and in a hand-written fixture would agree with
  // itself and with nothing else. Slot 0 holds the server's defaults, which
  // would prove the names and none of the values.
  const appSlot = slotsAfterSamplerRequest[3];
  const serverParams = appSlot.params as unknown as Record<string, number>;
  const serverSamplerNames = Object.keys(serverParams);

  const settings = {
    temperature: 0.33,
    top_p: 0.77,
    top_k: 11,
    min_p: 0.11,
    typical_p: 0.91,
    xtc_threshold: 0.31,
    xtc_probability: 0.21,
    penalty_last_n: 41,
    penalty_repeat: 1.11,
    penalty_freq: 0.41,
    penalty_present: 0.51,
    mirostat: 2,
    mirostat_tau: 4.1,
    mirostat_eta: 0.21,
    seed: 12345,
    n_predict: 128,
    n_probs: 3,
  };

  const BASE_KEYS = {
    temperature: 0.33,
    top_p: 0.77,
    max_completion_tokens: 128,
  };

  // Send name to the name `/slots` reports it back under. The two differ for
  // exactly one param, and the props read map is where that asymmetry is
  // already declared, so the check derives from it instead of naming the pair
  // again: a param whose send name the server does not know fails here even
  // though nobody added a case for it.
  const readNameOf = Object.fromEntries(
    Object.entries(llamaCpp.sendNames).map(([param, wireName]) => [
      wireName,
      PROPS_READ_NAMES[param as SamplerParam],
    ]),
  );

  it('spells every emitted sampler the way the server does', () => {
    const payload = samplerBody('llama.cpp', settings);

    // Every send name is exercised, so a param added to the map without a
    // value here fails rather than going unchecked.
    expect(Object.keys(payload).sort()).toEqual(
      Object.values(llamaCpp.sendNames).sort(),
    );
    for (const name of Object.keys(payload)) {
      expect(serverSamplerNames).toContain(readNameOf[name]);
    }
  });

  it('lands each value under the name the server reports it back under', () => {
    const payload = samplerBody('llama.cpp', settings);
    // The captured request set its own length and pre-dates n_probs
    // forwarding, so those two values cannot be read back from it.
    const notCaptured = ['n_predict', 'n_probs'];

    for (const [param, wireName] of Object.entries(llamaCpp.sendNames)) {
      if (notCaptured.includes(param)) {
        continue;
      }
      const readName = PROPS_READ_NAMES[param as SamplerParam];
      expect(serverParams[readName]).toBeCloseTo(Number(payload[wireName]), 6);
    }
  });

  it('renames the four penalties the server does not know by our names', () => {
    const ourPenaltyNames = [
      'penalty_last_n',
      'penalty_repeat',
      'penalty_freq',
      'penalty_present',
    ];
    for (const ours of ourPenaltyNames) {
      expect(serverSamplerNames).not.toContain(ours);
    }

    const payload = samplerBody('llama.cpp', settings);
    for (const ours of ourPenaltyNames) {
      expect(payload).not.toHaveProperty(ours);
    }
    expect(payload.repeat_last_n).toBe(41);
    expect(payload.repeat_penalty).toBe(1.11);
    expect(payload.frequency_penalty).toBe(0.41);
    expect(payload.presence_penalty).toBe(0.51);
  });

  it('forwards the allow-listed samplers and nothing else', () => {
    expect(samplerBody('llama.cpp', settings)).toEqual({
      ...BASE_KEYS,
      n_probs: 3,
      top_k: 11,
      min_p: 0.11,
      typical_p: 0.91,
      xtc_threshold: 0.31,
      xtc_probability: 0.21,
      repeat_last_n: 41,
      repeat_penalty: 1.11,
      frequency_penalty: 0.41,
      presence_penalty: 0.51,
      mirostat: 2,
      mirostat_tau: 4.1,
      mirostat_eta: 0.21,
      seed: 12345,
    });
  });

  it('omits a value that is not a finite number, and keeps a zero', () => {
    expect(
      samplerBody('llama.cpp', {
        top_k: undefined,
        min_p: NaN,
        typical_p: Infinity,
        mirostat: 0,
      }),
    ).toEqual({mirostat: 0});
  });

  it('sends only the base three to a server type with no send map of its own', () => {
    for (const serverType of [
      'vLLM',
      'Ollama',
      'OpenAI',
      'LM Studio',
      'something-else',
      '',
      undefined,
    ]) {
      expect(samplerBody(serverType, settings)).toEqual(BASE_KEYS);
    }
  });
});
