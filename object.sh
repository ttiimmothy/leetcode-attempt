#!/bin/bash
for i in {1..180}; do
  printf -v padded "%03d" "$i"  # Format the number with leading zeros
  # echo "object${padded}.mm"
  # echo "object${i}.mm"
  # echo "object\"${padded}\".mm"
  cp object.mm objectiveC++/object${padded}.mm
done