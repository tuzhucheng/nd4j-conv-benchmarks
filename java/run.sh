#!/usr/bin/env bash

set -x
mvn exec:java -Dexec.mainClass="com.example.app.App" -Dexec.args="$*"
