# def merge_queries(
#     db: DB,
#     queries: list[Query],
#     required: list[Query],
#     batch: int = 20,
# ):
#     # NOTE: in theory we should have a small (reducing) number of merged synthetics
#     # and only one layer of depth of non synthetic

#     required = list(sorted(required, key = lambda q: q.start))
#     queries = list(sorted(queries, key = lambda q: q.start))

#     if not len(queries):
#         queries = [
#             db.insert_query(required[0]._replace(
#                 id=None,
#                 synthetic=True
#             ))
#         ]

#     parents: list[int | None] = [None for _ in required]

#     for i, req in enumerate(required):
#         for j, q in enumerate(queries):
#             if q.start <= req.start and q.end >= req.start:
#                 parents[i] = q.id
#                 queries[j] = q._replace(
#                     end = max([q.end, req.end])
#                 )
#                 break
#         queries.append(db.insert_query(req._replace(
#             id=None,
#             synthetic=True,
#         )))
#         parents[i] = queries[-1].id

#     assert all([p is not None for p in parents]), parents

#     merged: list[int | None] = [None for _ in queries]

#     for i, q in enumerate(queries):
#         if i == 0:
#             continue
#         qp = queries[i-1]
#         if qp.end >= q.start:
#             par: int = (
#                 qp.id
#                 if merged[i-1] is None
#                 else merged[i-1] # type: ignore
#             )
#             merged[i] = par
#             queries[par] = queries[par]._replace(
#                 end=max([
#                     queries[par].end,
#                     q.end
#                 ])
#             )

#     q_par = {
#         q.id: par for q, par in zip(queries, merged)
#     }
#     parents = [
#         q_par[p] # type: ignore
#         for p in parents
#     ]

#     for r, rpar in zip(required, parents):
#         assert rpar is not None, (r, rpar)
#         db.insert_query(r._replace(parent_id=rpar))

#     res = []
#     for q, qpar in zip(queries, merged):

#         if qpar is None:
#             res.append(q)

#         q = q._replace(parent_id=qpar)
#         db.insert_query(q)

#         children = db.get_queries(
#             fields=None,
#             where=dict(
#                 parent_id=f"={q.id}",
#             )
#         )
#         for c in children:
#             db.insert_query(c._replace(parent_id=qpar))

#     for r in reversed(required):
#         if r.expected < batch:
#             lhs = r.start
#             prev = db.get_queries(
#                 fields=None,
#                 where=dict(
#                     start=f"={lhs.toordinal()}",
#                     synthetic="=0",
#                 )
#                 one=True,
#             )
#             while prev is not None:

#     return res

# required = sum([
#     db.get_queries(
#         fields=dict(id=int),
#         where=dict(
#             parent_id=f"={q.id}",
#             synthetic=f"=0",
#         )
#     )
#     for q in queries
# ], [])
# q_ids = [r["id"] for r in required]
# # TODO: this will possibly return dup data around the query boundaries?
# return db.get_bars(
#     fields=None,
#     where=dict(
#         query_id=f" IN {q_ids}",
#         contract_id=f"={contract.id}",
#         date = f" BETWEEN {start.toordinal()} AND {end.toordinal()}"
#         # TODO: this conversion like bool should behandled by sql helper module
#     ),
#     order=dict(date="ASC")
# )
