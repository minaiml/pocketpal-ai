import type {ReasoningIntent} from '../../utils/completionTypes';
import {reasoningBudgetFor} from '../../utils/reasoningCapability';
import {BASE_SEND_NAMES, openAICompatible, sendSamplers} from './base';
import type {ServerDialect} from './dialect';
import {readLlamaCppListRow} from './llamaCppListRow';

/**
 * The four `penalty_*` renames are the wire's names, not ours: llama-server
 * accepts an unknown key with a 200 and ignores it, so under our own spelling
 * the sampler silently keeps its default.
 */
const sendNames = {
  ...BASE_SEND_NAMES,
  top_k: 'top_k',
  min_p: 'min_p',
  typical_p: 'typical_p',
  xtc_threshold: 'xtc_threshold',
  xtc_probability: 'xtc_probability',
  penalty_last_n: 'repeat_last_n',
  penalty_repeat: 'repeat_penalty',
  penalty_freq: 'frequency_penalty',
  penalty_present: 'presence_penalty',
  mirostat: 'mirostat',
  mirostat_tau: 'mirostat_tau',
  mirostat_eta: 'mirostat_eta',
  seed: 'seed',
  n_probs: 'n_probs',
} as const;

/**
 * `reasoning_format` is always `'auto'`: a no-op for non-reasoning models and
 * the value that extracts reasoning into `reasoning_content` instead of leaking
 * raw channel/think markers into content (e.g. gemma-4 emits an empty
 * `<|channel>thought` block even when thinking is off). On/off is carried
 * solely by `enable_thinking`. An effort that is not a level sends no budget,
 * which leaves the server's own default in force.
 */
function reasoningExtras(
  reasoning: ReasoningIntent | undefined,
): Record<string, unknown> {
  if (!reasoning) {
    return {};
  }
  const {enabled, effort} = reasoning;
  if (!enabled) {
    return {
      reasoning_format: 'auto',
      chat_template_kwargs: {enable_thinking: false},
    };
  }
  if (!effort) {
    return {reasoning_format: 'auto'};
  }
  const budget = reasoningBudgetFor(effort);
  return {
    reasoning_format: 'auto',
    chat_template_kwargs: {reasoning_effort: effort},
    ...(budget !== undefined && {reasoning_budget_tokens: budget}),
  };
}

export const llamaCpp = {
  ...openAICompatible,
  type: 'llama.cpp',
  sendNames,
  bodyExtras: ({samplers, reasoning}) => ({
    ...sendSamplers(sendNames, samplers),
    ...reasoningExtras(reasoning),
  }),
  readModelEntry: readLlamaCppListRow,
  discovery: {
    hasProps: true,
    listReportsCaps: true,
  },
} satisfies ServerDialect;
