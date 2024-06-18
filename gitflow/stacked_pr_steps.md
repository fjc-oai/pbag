1. work on feature branch 1
    - git checkout -b fjc/feature-1
    - make code change
    - git add -A && git commit -a -m 'feature 1'
    - git push
2. create pr on github
3. work on feature branch 2
    - git checkout -b fjc/feature-2
    - make code change
    - git add -A && git commit -a -m 'feature 2'
    - git push
4. create pr-2 on github
    - switch base branch from master to feature-1 to check diff of feature-2
    - pr title will change
        - fmars wants to merge 2 commits into master from fjc/feature-2
        -> fmars wants to merge 2 commits into fjc/feature-1 from fjc/feature-2
5. continue working on pr-2, while merging pr-1
6. once pr-1 is merged, update pr-2
    - git checkout master && git fetch --prune && git rebase origin/master
    - git checkout fjc/feature-2
    - git merge master
    - git push
7. merge pr-2 