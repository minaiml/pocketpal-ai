import type {
  ListDerivedCaps,
  RemoteModelInfo,
  ServerConfig,
} from '../../utils/types';
import {readLlamaCppListRow} from './llamaCppListRow';

/**
 * Read a models-list row for capabilities. Pure, and never a default: anything
 * absent, wrongly typed or unparseable yields no field at all, so a caller can
 * tell "this server says no" from "this server did not say".
 *
 * The `serverType` gate lives here rather than at the call sites because the
 * two callers source that type differently, and a gate outside the function is
 * a gate that can disagree with itself.
 */
export function deriveListCaps(
  row: RemoteModelInfo | undefined,
  serverType: string | undefined,
): ListDerivedCaps {
  if (serverType !== 'llama.cpp') {
    return {tier: 'list'};
  }
  return readLlamaCppListRow(row);
}

/**
 * The same derivation for every model of every server, keyed as
 * `${serverId}/${remoteModelId}` — the id a remote `Model` carries.
 */
export function deriveListCapsMap(
  servers: ServerConfig[],
  serverModels: {get(serverId: string): RemoteModelInfo[] | undefined},
): Record<string, ListDerivedCaps> {
  const map: Record<string, ListDerivedCaps> = {};
  for (const server of servers) {
    for (const row of serverModels.get(server.id) ?? []) {
      map[`${server.id}/${row.id}`] = deriveListCaps(row, server.serverType);
    }
  }
  return map;
}
