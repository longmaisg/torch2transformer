#!/bin/bash

git init
git add .
git commit -m "Initial torch2transformer scaffold"
git branch -M main

git remote add origin git@github.com:longmaisg/torch2transformer.git
git remote -v

git push -u origin main

