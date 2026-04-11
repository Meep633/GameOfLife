#!/bin/bash

mpirun --bind-to core -np $1 ./conway $2 64