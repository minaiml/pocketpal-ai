import {DIALECTS, dialectFor} from '../index';
import {openAICompatible} from '../base';
import type {DialectRequest, ServerDialect} from '../dialect';
import type {ServerType} from '../../../utils/serverTypes';
import {streamFinishChunk} from '../../../../jest/fixtures/llamaServerWire';

// The keys the transport writes itself. One returned by a dialect would be
// silently overwritten, so no dialect may name one.
const TRANSPORT_BODY_KEYS = [
  'model',
  'messages',
  'stream',
  'stop',
  'tools',
  'tool_choice',
  'response_format',
];

const SAMPLERS = {
  temperature: 0.7,
  top_p: 0.9,
  top_k: 10,
  min_p: 0.05,
  typical_p: 0.95,
  xtc_threshold: 0.1,
  xtc_probability: 0.2,
  penalty_last_n: 64,
  penalty_repeat: 1.2,
  penalty_freq: 0.4,
  penalty_present: 0.5,
  mirostat: 2,
  mirostat_tau: 5,
  mirostat_eta: 0.1,
  seed: 42,
  n_predict: 1024,
};

const REQUESTS: Array<[string, DialectRequest]> = [
  ['no samplers, no reasoning', {samplers: {}}],
  ['every sampler', {samplers: SAMPLERS}],
  ['reasoning off', {samplers: SAMPLERS, reasoning: {enabled: false}}],
  ['reasoning on', {samplers: SAMPLERS, reasoning: {enabled: true}}],
  [
    'reasoning on with a level',
    {samplers: SAMPLERS, reasoning: {enabled: true, effort: 'high'}},
  ],
  [
    'reasoning on with a non-level',
    {samplers: SAMPLERS, reasoning: {enabled: true, effort: 'extreme'}},
  ],
];

const entries = Object.entries(DIALECTS) as Array<[ServerType, ServerDialect]>;

describe('DIALECTS', () => {
  it.each(entries)('%s is keyed by its own type', (type, dialect) => {
    expect(dialect.type).toBe(type);
  });

  it.each(['', undefined, 'Llama.CPP', 'my server', 42])(
    'resolves the persisted value %p to the base',
    raw => {
      expect(dialectFor(raw)).toBe(openAICompatible);
    },
  );

  it('resolves a known type to its own dialect', () => {
    expect(dialectFor('llama.cpp')).toBe(DIALECTS['llama.cpp']);
  });
});

describe.each(entries)('the %s dialect', (_type, dialect) => {
  it.each(REQUESTS)('returns no transport-owned key for %s', (_name, req) => {
    for (const key of Object.keys(dialect.bodyExtras(req))) {
      expect(TRANSPORT_BODY_KEYS).not.toContain(key);
    }
  });

  it.each(REQUESTS)('assembles the same body twice for %s', (_name, req) => {
    const unchanged = JSON.stringify(req);

    expect(dialect.bodyExtras(req)).toEqual(dialect.bodyExtras(req));
    expect(JSON.stringify(req)).toBe(unchanged);
  });

  it('forwards exactly the names in its own send map', () => {
    const body = dialect.bodyExtras({samplers: SAMPLERS});

    expect(Object.keys(body).sort()).toEqual(
      Object.values(dialect.sendNames).sort(),
    );
  });

  it('omits a non-finite sampler rather than coercing it', () => {
    const body = dialect.bodyExtras({
      samplers: {temperature: NaN, top_p: Infinity, n_predict: NaN},
    });

    expect(body).toEqual({});
  });

  it.each([undefined, null, 42, 'nope', {}, {timings: 'none'}])(
    'reads %p as no finish facts',
    chunk => {
      expect(dialect.readFinish(chunk)).toEqual({});
    },
  );

  it('sums the evaluated and cached prompt tokens off a verbatim chunk', () => {
    expect(dialect.readFinish(streamFinishChunk)).toEqual({
      timings: streamFinishChunk.timings,
      tokensEvaluated: 16,
      tokensPredicted: 6,
    });
  });
});
