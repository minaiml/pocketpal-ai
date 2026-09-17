import type {ReasoningIntent} from '../../utils/completionTypes';
import {BASE_SEND_NAMES, openAICompatible, sendMap} from './base';
import type {ServerDialect} from './dialect';

/**
 * Modern vLLM ignores an unknown `chat_template_kwargs` entry, so both keys are
 * safe. Its sampler names are its own (`repetition_penalty`, not
 * `repeat_penalty`), so none is forwarded until one has been read back off a
 * live server.
 */
function reasoningExtras(
  reasoning: ReasoningIntent | undefined,
): Record<string, unknown> {
  if (!reasoning) {
    return {};
  }
  const {enabled, effort} = reasoning;
  if (!enabled) {
    return {chat_template_kwargs: {enable_thinking: false}};
  }
  return effort ? {chat_template_kwargs: {reasoning_effort: effort}} : {};
}

export const vllm = {
  ...openAICompatible,
  type: 'vLLM',
  ...sendMap(BASE_SEND_NAMES, reasoningExtras),
} satisfies ServerDialect;
