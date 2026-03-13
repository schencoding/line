#ifndef _NANO_H
#define _NANO_H

#include <bitset>
#include <cassert>
#include <cstdint>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <emmintrin.h>
#include <immintrin.h>

thread_local unsigned long long search_in_nano_num = 0;

#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE	(64)
#endif

// Set nano to default: 256B, 4 cache lines
#define NANO_KEY_NUM        15
#define NANO_LINE_NUM       4    // 256B
#define NANO_INIT_KEY_NUM   12

#define bitScan(x)  __builtin_ffs(x)
#define countBit(x) __builtin_popcount(x)

static int last_slot_in_line[NANO_KEY_NUM] = {2,2,2,6,6,6,6,10,10,10,10,14,14,14,14};

/********PREFETCH********/

#define prefetcht0(mem_var)     \
        __asm__ __volatile__ ("prefetcht0 %0": :"m"(mem_var))
#define pref(mem_var)      prefetcht0(mem_var)

#define prefetcht2(mem_var)     \
        __asm__ __volatile__ ("prefetcht2 %0": :"m"(mem_var))
#define preft2(mem_var)      prefetcht2(mem_var)

static void inline NANO_PREF(void *bbp)
{
    pref (* ((char *)bbp));
    pref (* ((char *)bbp + CACHE_LINE_SIZE));
    pref (* ((char *)bbp + CACHE_LINE_SIZE*2));
    pref (* ((char *)bbp + CACHE_LINE_SIZE*3));
}

static void inline NANO_PREF_T2(void *bbp) {
    preft2 (* ((char *)bbp));
    preft2 (* ((char *)bbp + CACHE_LINE_SIZE));
    preft2 (* ((char *)bbp + CACHE_LINE_SIZE*2));
    preft2 (* ((char *)bbp + CACHE_LINE_SIZE*3));
}

/******PREFETCH END******/

/**
 * nano node
 * 256 Byte when both K and VA are 8 Byte
 */
template <class K, class VA>
class nano {
public:
    // type define begin
    typedef nano<K, VA> self_type;
    typedef std::pair<K, VA> PA;

    typedef union nanoMeta {
        unsigned long long  word8B[2];
        struct {
            uint16_t         bitmap     :15;
            uint16_t         overflow   :1;
            unsigned char    fgpt[14];  /* fingerprints */
        } v;
    } nanoMeta;

    typedef struct IdxEntry {
        K   k;
        VA  ch;
    } IdxEntry;
    // type define end

    static inline unsigned char hashcode1B(K x) {
        uint64_t llx = (unsigned long long)x;
        llx ^= llx>>32;
        llx ^= llx>>16;
        llx ^= llx>>8;
        return (unsigned char)(llx&0x0ffULL);
    }

    uint16_t            bitmap  :15;
    uint16_t            overflow:1;
    unsigned char       fgpt[14]; /* fingerprints */
    IdxEntry            ent[15];

public:
    K & k(int idx)  { return ent[idx].k; }
    K ck(int idx) const { return ent[idx].k; }
    VA & ch(int idx) { return ent[idx].ch; }
    VA cch(int idx) const { return ent[idx].ch; }

    int exist(int idx) { return (bitmap) & (0x1<<idx); }

    int num() const {return countBit(bitmap);}

    bool isFull(void)  { return (bitmap == 0x7fff); }
    bool isEmpty(void) { return (bitmap == 0x0000); }

    void setOverflow(void) {
        this->overflow = 1;
    }

    int getOverflow(void) {
        return overflow;
    }

    void setAllWords(nanoMeta *m) {
       nanoMeta * my_meta= (nanoMeta *)this;
       my_meta->word8B[1]= m->word8B[1];
       my_meta->word8B[0]= m->word8B[0];
    }

    void setWord0(nanoMeta *m) {
       nanoMeta * my_meta= (nanoMeta *)this;
       my_meta->word8B[0]= m->word8B[0];
    }

    K getMinKey() const {
        K min_key = std::numeric_limits<K>::max();
        uint16_t bm = this->bitmap;

        while (bm) {
            int jj = bitScan(bm)-1;  // next candidate
            if (this->ck(jj) < min_key) { // larger
                min_key = this->ck(jj);
            }
            bm &= ~(0x1<<jj);  // remove this bit
        } // end while
        return min_key;
    }

    K getLargestKey() const {
        K max_key = std::numeric_limits<K>::lowest();
        uint16_t bm = this->bitmap;

        while (bm) {
            int jj = bitScan(bm)-1;  // next candidate
            if (this->ck(jj) > max_key) { // larger
                max_key = this->ck(jj);
            }
            bm &= ~(0x1<<jj);  // remove this bit
        } // end while
        return max_key;
    }

    void printNano() const {
        std::cout << "--------------" << std::endl;
        std::cout << "nano key num: " << this->num() << std::endl;
        uint16_t bm = this->bitmap;
        while (bm) {
            int jj = bitScan(bm)-1;  // next candidate
            std::cout << "key: " << (this->ck(jj)) << ", value: " << (this->cch(jj)) << std::endl;
            std::cout << "Fingerprint: " << static_cast<int>(hashcode1B(this->ck(jj))) << std::endl;
            bm &= ~(0x1<<jj);  // remove this bit
        } // end while
    }

    void prefetchNano() {
        NANO_PREF(this);
    }

    int searchInNano(const K& key) {
        search_in_nano_num++;
        unsigned char key_hash= hashcode1B(key);
        // SIMD comparison
        // a. set every byte to key_hash in a 16B 
        __m128i key_16B = _mm_set1_epi8((char)key_hash);
        // b. load meta into another 16B register
        __m128i fgpt_16B= _mm_load_si128((const __m128i*)this);
        // c. compare them
        __m128i cmp_res = _mm_cmpeq_epi8(key_16B, fgpt_16B);
        // d. generate a mask
        unsigned int mask= (unsigned int)
                            _mm_movemask_epi8(cmp_res);  // 1: same; 0: diff
        // remove the lower 2 bit and set 14th bit then AND bitmap
        mask=  ((mask >> 2) | (0x1 << 14)) & ((unsigned int)(this->bitmap));

        // search every matching candidate
        while (mask) {
            int jj = bitScan(mask)-1;  // next candidate
            if (this->k(jj) == key) {  // found
                return jj;
            }
            mask &= ~(0x1<<jj);        // remove this bit
        } // end while

        return -1;
    } // end of searchInNano

    int insertInNanoWithoutMoving(const K& key, const VA& ptr) {
        unsigned char key_hash= hashcode1B(key);

        // SIMD comparison
        __m128i key_16B = _mm_set1_epi8((char)key_hash);
        __m128i fgpt_16B= _mm_load_si128((const __m128i*)this);
        __m128i cmp_res = _mm_cmpeq_epi8(key_16B, fgpt_16B);
        unsigned int mask= (unsigned int)
                            _mm_movemask_epi8(cmp_res);  // 1: same; 0: diff
        mask=  ((mask >> 2) | (0x1 << 14)) & ((unsigned int)(this->bitmap));

        // search every matching candidate
        while (mask) {
            int jj = bitScan(mask)-1;  // next candidate
            if (this->k(jj) == key) {  // found: do nothing, return
                return -2;
            }
            mask &= ~(0x1<<jj);  // remove this bit
        } // end while

        nanoMeta meta= *((nanoMeta *)this);

        /* nano is not full */
        if (!this->isFull()) {
            // 1. get first empty slot
            uint16_t bitmap= meta.v.bitmap;
            int slot= bitScan(~bitmap)-1;
            // 2. set nano.entry[slot]= (k, v);
            //    set fgpt, bitmap in meta
            this->k(slot)= key;
            this->ch(slot)= ptr;
            if (slot < 14) {
                meta.v.fgpt[slot]= key_hash;
            }
            bitmap |= (1<<slot); 
            meta.v.bitmap= bitmap;
            this->setAllWords(&meta);
            return slot;
        } // end of not full
        else { // full
            return -1;
        } // end of full
    } // end of insertInNanoWithoutMoving

    void deleteInNano(const K& key, int key_idx) {
        // set bitmap
        nanoMeta meta= *((nanoMeta *)this);
        meta.v.bitmap &= ~(1<<key_idx);  // mark the bitmap to delete the entry
        this->setWord0(&meta);
    }  // end of deleteInNano

    void updateInNano(const K& key, const VA& ptr, int key_idx) {
        // set the value
        this->ch(key_idx) = ptr;
    }  // end of updateInNano
}; // end of class nano

#endif