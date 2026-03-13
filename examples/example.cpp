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
