# TODO: Writer

# in theory flush could even return a function, that queries the graph for the data

# so we don't hog up memory, only have to realise when we then call files.csv.write(acc)
# which opens the file, iters the acc, calls, immediately writes, and then continues
