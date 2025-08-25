#!/bin/bash
NEW_URL=$1
echo "{\"api\":{\"url\":\"$NEW_URL/openapi.yaml\"}}" > .well-known/ai-plugin.json
echo "openapi: 3.0.1
servers:
  - url: $NEW_URL" > openapi.yaml

