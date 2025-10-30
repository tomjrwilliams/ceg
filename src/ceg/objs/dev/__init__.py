
# maybe organise d001_{...}, d002_{...} etc.

# possibly under the domain names, then prod is just under domain names

# ie the d001 etc. can appear in object paths, and then the file system itself gives you a more full name (then the tag can have the long form name)




# prod is just configs for objects with params, dependencies, etc.

# all the calcs should be from main libraries




# dev is a staging area for work in progress nodes, etc.

# you copy in what you want to change, edit it, appopriately identified so it wont override prod

# then when you're done, you copy it back into the relevant place in the main repo



# so dev is project based: each folder is a project, that when deployed to prod is deleted (from master), but still in history, at the point the code is moved into the appropriate main library module



# and so dev should pull in prod / other dev projects, and then override with whatever you're working on for that project


