import {normaliseTimings} from '../../utils/completionTypes';
import type {ReasoningIntent} from '../../utils/completionTypes';
import {finiteNumber} from '../../utils/finite';
import type {SamplerParam, Samplers} from '../../utils/samplerParams';
import type {DialectRequest, FinishRead, ServerDialect} from './dialect';

/**
 * The three names every OpenAI-compatible server answers to. `n_predict` is
 * ours; `max_completion_tokens` is the wire's.
 */
export const BASE_SEND_NAMES = {
  temperature: 'temperature',
  top_p: 'top_p',
  n_predict: 'max_completion_tokens',
} as const;

/**
 * `sendNames[p]: samplers[p]` for every forwarded param holding a finite
 * number. A missing or non-finite value is omitted, never coerced: a server
 * reading JSON `null` for a sampler either 400s or silently defaults.
 */
export function sendSamplers(
  sendNames: Partial<Record<SamplerParam, string>>,
  samplers: Samplers,
): Record<string, number> {
  const body: Record<string, number> = {};
  for (const [param, wireName] of Object.entries(sendNames)) {
    const value = finiteNumber(samplers[param as SamplerParam]);
    if (value !== undefined) {
      body[wireName] = value;
    }
  }
  return body;
}

/**
 * Timings off a final chunk, plus the token counts derived from them. The
 * server evaluates only the prompt tokens it did not already hold in its KV
 * cache, so the prompt total is `prompt_n + cache_n`, each key guarded on its
 * own: a build too old to report reuse omits `cache_n`, while a cold prompt on
 * a newer one reports 0, and those are different facts.
 */
export function readTimingsFinish(chunk: unknown): FinishRead {
  const timings = normaliseTimings(
    (chunk as {timings?: unknown} | null | undefined)?.timings,
  );
  if (!timings) {
    return {};
  }
  const tokensEvaluated =
    timings.prompt_n !== undefined || timings.cache_n !== undefined
      ? (timings.prompt_n ?? 0) + (timings.cache_n ?? 0)
      : undefined;
  return {timings, tokensEvaluated, tokensPredicted: timings.predicted_n};
}

/**
 * A dialect's send map together with the body that emits it, as one statement:
 * naming a sampler is what forwards it, so the two cannot come to disagree.
 * The generic keeps the caller's literal type, which is what makes the
 * llama.cpp map total over `SamplerParam` where the props reader spreads it.
 */
export function sendMap<T extends Partial<Record<SamplerParam, string>>>(
  sendNames: T,
  reasoningExtras: (
    reasoning: ReasoningIntent | undefined,
  ) => Record<string, unknown> = () => ({}),
): {
  sendNames: T;
  bodyExtras: (req: DialectRequest) => Record<string, unknown>;
} {
  return {
    sendNames,
    bodyExtras: ({samplers, reasoning}) => ({
      ...sendSamplers(sendNames, samplers),
      ...reasoningExtras(reasoning),
    }),
  };
}

/** What every server type receives, `'unknown'` included. */
export const openAICompatible = {
  type: 'unknown',
  ...sendMap(BASE_SEND_NAMES),
  readFinish: readTimingsFinish,
  readModelEntry: () => ({tier: 'list'}),
  discovery: {
    hasProps: false,
    listReportsCaps: false,
  },
} satisfies ServerDialect;
