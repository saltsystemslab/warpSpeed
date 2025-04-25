#ifndef HT_PAIRS
#define HT_PAIRS


#include <gallatin/allocators/alloc_utils.cuh>

namespace warpSpeed {

namespace tables {

  struct packed_tags {
    uint64_t first;
    uint64_t second;
  };

  template <typename Key, typename Val>
   struct ht_pair{
      Key key;
      Val val;
   };


 }

}



#endif