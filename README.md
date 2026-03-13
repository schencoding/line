# LINE

Thanks for your interest in our work "LINE: A Learned Index with Group-Enhanced Leaves and Cache-Optimized Inner Tree"

This project contains the code of LINE.

## Compile & Run

```bash
mkdir build
cd build
cmake ..
make
```

or simply
```bash
./build.sh
```

Run example:

```bash
./build/example
```

## Macro Setting

Please set **ALEX_USE_LZCNT** to 0 in `line/core/alex_src/alex_nodes.h` and `line/core/alexol_src/alex_nodes.h` if your hardware does not support lzcnt/tzcnt. 

Uncomment **#define USING_LOCK** in `line/core/line_nodes.h` and set **INNER** to 3 in `line/core/line.h` to support multi-threaded operations. ALEXol relies on TBB to run. 

## Usage

`examples/example.cpp` demonstrates the usage of LINE:
```cpp
#include <iostream>
#include "../core/line.h"

using namespace std;

int main() {
  line::Line<unsigned long, unsigned long, false> index;
  int key_num = 1000;
  pair<unsigned long, unsigned long> *keys = new pair<unsigned long, unsigned long>[key_num];
  for (int i = 0; i < 1000; i++) {
    keys[i]={2*i,2*i};
  }

  index.bulk_load(keys, 1000);
  
  for (int i = 0; i < 1000; i++) {
    index.insert(2*i+1, 2*i+1);
  }

  for (int i = 0; i < 2000; i++) {
    auto res = index.get_payload(i);
    if (res) {
        cout<<"value at " << i << ": " << *res <<std::endl;
    }
  }

  return 0;
}

```
