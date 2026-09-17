import type {CompletionParams} from './completionTypes';
import {finiteNumber} from './finite';

/**
 * Every numeric completion control the app can forward, under the app's own
 * names. A wire name belongs to a dialect's send map, never here.
 */
export const SAMPLER_PARAMS = [
  'temperature',
  'top_p',
  'top_k',
  'min_p',
  'typical_p',
  'xtc_threshold',
  'xtc_probability',
  'penalty_last_n',
  'penalty_repeat',
  'penalty_freq',
  'penalty_present',
  'mirostat',
  'mirostat_tau',
  'mirostat_eta',
  'seed',
  'n_predict',
  'n_probs',
] as const satisfies readonly (keyof CompletionParams)[];

export type SamplerParam = (typeof SAMPLER_PARAMS)[number];

export type Samplers = Partial<Record<SamplerParam, number>>;

export function pickSamplers(params: CompletionParams): Samplers {
  const samplers: Samplers = {};
  for (const param of SAMPLER_PARAMS) {
    const value = finiteNumber(params[param]);
    if (value !== undefined) {
      samplers[param] = value;
    }
  }
  return samplers;
}
