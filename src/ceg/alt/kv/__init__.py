
# backends determine permissive i types, what goes in
# I is fixed per backend (store)

# front end determines o types, map to i (both key and value)
# o ideally is broad as possible: np, polars, etc.

# o is business logic, needs to implement codign for whatever output (business) types you have, given the input stores you intend to use