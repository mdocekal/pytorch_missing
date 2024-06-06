#pragma once

#define THREADS 256
#define BLOCKS(N) (N + THREADS - 1) / THREADS