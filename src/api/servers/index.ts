import {toServerType} from '../../utils/serverTypes';
import type {ServerType} from '../../utils/serverTypes';
import {openAICompatible} from './base';
import type {ServerDialect} from './dialect';
import {llamaCpp} from './llamaCpp';
import {lmStudio} from './lmStudio';
import {ollama} from './ollama';
import {openaiPlatform} from './openaiPlatform';
import {vllm} from './vllm';

/**
 * One dialect per server type. Total over `ServerType`, so a new type cannot be
 * offered in the UI before it has a dialect.
 */
export const DIALECTS: Record<ServerType, ServerDialect> = {
  'llama.cpp': llamaCpp,
  'LM Studio': lmStudio,
  Ollama: ollama,
  OpenAI: openaiPlatform,
  vLLM: vllm,
  unknown: openAICompatible,
};

/**
 * The dialect for a persisted type. The parameter admits the legacy shapes a
 * stored row can still hold — an empty string, a free string, a case variant,
 * an absent key — and each of those speaks the base, which is what they reach
 * today. A value of another kind is a mistake, not a legacy case, so it is a
 * compile error rather than a silent base dialect.
 */
export function dialectFor(
  raw: ServerType | string | undefined,
): ServerDialect {
  return DIALECTS[toServerType(raw)];
}
