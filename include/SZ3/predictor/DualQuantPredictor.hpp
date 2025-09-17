#ifndef _SZ_DUALQUANT_HPP
#define _SZ_DUALQUANT_HPP

#include <cassert>
#include <experimental/simd>
#include <cmath> 
#include <cfenv>
#pragma STDC FENV_ACCESS ON

#include <iostream>

#include "SZ3/def.hpp"
#include "SZ3/predictor/Predictor.hpp"
#include "SZ3/utils/Iterator.hpp"


namespace stdx {
    using namespace std::experimental;
    using namespace std::experimental::__proposed;
}

namespace SZ3 {

// N-dimension L-layer dualquant predictor
template <class T, uint N, uint L>
class DualQuantPredictor : public concepts::PredictorInterface<T, N> {
   public:
    static const uint8_t predictor_id = 0b00000100;
    using Range = multi_dimensional_range<T, N>;
    using iterator = typename multi_dimensional_range<T, N>::iterator;

    DualQuantPredictor() { this->noise = 0; }

    DualQuantPredictor(double eb) {
        this->noise = 0;
        this->eb = eb;
        ebs_x2r = 1 / (2 * eb);
        if (L == 1) {
            if (N == 1) {
                this->noise = 0.5 * eb;
            } else if (N == 2) {
                this->noise = 0.81 * eb;
            } else if (N == 3) {
                this->noise = 1.22 * eb;
            } else if (N == 4) {
                this->noise = 1.79 * eb;
            }
        } else if (L == 2) {
            if (N == 1) {
                this->noise = 1.08 * eb;
            } else if (N == 2) {
                this->noise = 2.76 * eb;
            } else if (N == 3) {
                this->noise = 6.8 * eb;
            }
        }
    }

    void precompress_data(const iterator &) const override {}

    void postcompress_data(const iterator &) const override {}

    void predecompress_data(const iterator &) const override {}

    void postdecompress_data(const iterator &) const override {}

    bool precompress_block(const std::shared_ptr<Range> &element_range) override {
        
        const size_t batch_size = stdx::native_simd<T>::size();
        auto element = element_range->begin();
        auto end = element_range->end();
        auto cols = element.get_dimensions().back();

        const size_t full_batches = cols/ batch_size;
        const size_t remainder = cols % batch_size;

        while(element != end){
            for(size_t b = 0; b < full_batches; ++b){
                prequant(element);
                element += batch_size;
            }
            for(size_t r = 0; r < remainder; ++r){
                prequant_sequential(element);
                ++element;
            }
        }
        return true;
    }

    void precompress_block_commit() noexcept override {}

    bool predecompress_block(const std::shared_ptr<Range> &) override {return true;}

    /*
     * save doesn't need to store anything except the id
     */
    void save(uchar *&c) const override {
        c[0] = predictor_id;
        c += sizeof(uint8_t);

        *reinterpret_cast<size_t *>(c) = unpred_from_rounding_value.size();
        c += sizeof(size_t);
        memcpy(c, unpred_from_rounding_value.data(), unpred_from_rounding_value.size() * sizeof(T));
        c += unpred_from_rounding_value.size() * sizeof(T);

        *reinterpret_cast<size_t *>(c) = unpred_from_rounding_index.size();
        c += sizeof(size_t);
        memcpy(c, unpred_from_rounding_index.data(), unpred_from_rounding_index.size() * sizeof(uint64_t));
        c += unpred_from_rounding_index.size() * sizeof(uint64_t);

    }

    /*
     * just verifies the ID, increments
     */
    // static LorenzoPredictor<T,N> load(const unsigned char*& c, size_t& remaining_length) {
    //   assert(remaining_length > sizeof(uint8_t));
    //   c += 1;
    //   remaining_length -= sizeof(uint8_t);
    //   return LorenzoPredictor<T,N>{};
    // }
    void load(const uchar *&c, size_t &remaining_length) override {
        c += sizeof(uint8_t);
        remaining_length -= sizeof(uint8_t);

        size_t unpred_size = *reinterpret_cast<const size_t *>(c);
        c += sizeof(size_t);
        remaining_length -= sizeof(size_t);
        this->unpred_from_rounding_value = std::vector<T>(reinterpret_cast<const T *>(c), reinterpret_cast<const T *>(c) + unpred_size);
        c += unpred_size * sizeof(T);
        remaining_length -= unpred_size * sizeof(T);

        size_t unpred_size2 = *reinterpret_cast<const size_t *>(c);
        c += sizeof(size_t);
        remaining_length -= sizeof(size_t);
        this->unpred_from_rounding_index = std::vector<uint64_t>(reinterpret_cast<const uint64_t *>(c), reinterpret_cast<const uint64_t *>(c) + unpred_size2);
        c += unpred_size2 * sizeof(uint64_t);
        remaining_length -= unpred_size2 * sizeof(uint64_t);


    }

    void print() const override {
        std::cout << L << "-Layer " << N << "D Lorenzo predictor, noise = " << noise << "\n";
    }

    inline T estimate_error(const iterator &iter) const noexcept override {
        // return fabs(*iter - predict(iter)) + this->noise;
        return 0;
    }

    inline T predict(const iterator &iter) const noexcept override {
        return do_predict(iter);
    }

    
    inline stdx::native_simd<T> simd_predict(const iterator &iter) const noexcept {
        return do_simdpredict(iter);
    }
    
    inline std::vector<T>& get_unpred_value(){
        return unpred_from_rounding_value;
    }
    inline std::vector<uint64_t>& get_unpred_index(){
        return unpred_from_rounding_index;
    }

    size_t size_est() { return unpred_from_rounding_index.size() * sizeof(uint64_t) + unpred_from_rounding_value.size() * sizeof(T); }

   protected:
    T noise = 0;

   private:
    // variables
    double eb;
    double ebs_x2r;

    std::vector<uint64_t> unpred_from_rounding_index;
    std::vector<T> unpred_from_rounding_value;
    
    #pragma GCC push_options
    #pragma GCC optimize ("O1")
    inline void prequant(auto &iter) {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>){
            stdx::native_simd<T> simd_vector;
            simd_vector.copy_from(&(*iter), stdx::element_aligned);
            stdx::native_simd<T> multipler = static_cast<T>(ebs_x2r);
            std::fesetround(FE_TONEAREST);
            stdx::native_simd<T> temp_vector;
            temp_vector = stdx::nearbyint(simd_vector * multipler);

            stdx::native_simd<T> eb_x2 =  2 * static_cast<T>(eb);
            stdx::native_simd<T> temp_vector_2 = temp_vector * eb_x2;
            stdx::native_simd<T> difference = simd_vector - temp_vector_2;
            auto offset = iter.get_offset();

            for(std::size_t i = 0; i < simd_vector.size(); ++i){
                if(std::fabs(difference[i]) > eb){
                    unpred_from_rounding_index.push_back(offset + i);
                    unpred_from_rounding_value.push_back(simd_vector[i]);
                }      
            }
            temp_vector.copy_to(&(*iter), stdx::element_aligned);
        }else{
            stdx::native_simd<T> simd_vector;
            simd_vector.copy_from(&(*iter), stdx::element_aligned);
            stdx::native_simd<T> multipler = static_cast<T>(ebs_x2r);
            std::fesetround(FE_TONEAREST);
            stdx::native_simd<T> temp_vector;
            temp_vector = simd_vector * multipler;

            stdx::native_simd<T> eb_x2 =  2 * static_cast<T>(eb);
            stdx::native_simd<T> temp_vector_2 = temp_vector * eb_x2;
            stdx::native_simd<T> difference = simd_vector - temp_vector_2;
            auto offset = iter.get_offset();

            for(std::size_t i = 0; i < simd_vector.size(); ++i){
                if(std::abs(difference[i]) > eb){
                    unpred_from_rounding_index.push_back(offset + i);
                    unpred_from_rounding_value.push_back(simd_vector[i]);
                }
            }
            temp_vector.copy_to(&(*iter), stdx::element_aligned);
        }
    }
    #pragma GCC pop_options

    inline void prequant_sequential(auto &iter) {
        std::fesetround(FE_TONEAREST);
        auto temp_value = *iter * ebs_x2r;
        auto temp_value_2 = temp_value * 2 *eb;
        auto difference = *iter - temp_value_2;
        if(std::fabs(difference) > eb){
            unpred_from_rounding_index.push_back(iter.get_offset());
            unpred_from_rounding_value.push_back(*iter);
        }
        *iter = std::nearbyint(temp_value);
    }


    inline stdx::native_simd<T> addr_loc(T* address, bool &out_of_bound) const {
        stdx::native_simd<T> temp_vector;
        if(address == NULL){
            temp_vector = 0;
            return temp_vector;
        }else if(out_of_bound == true){
            temp_vector.copy_from(address, stdx::element_aligned);
            temp_vector[0] = 0;
            out_of_bound = false;
            return temp_vector;
        }else{
            temp_vector.copy_from(address, stdx::element_aligned);
            return temp_vector;
        }

    }

    template <uint NN = N, uint LL = L>
    inline typename std::enable_if<NN == 1 && LL == 1, stdx::native_simd<T>>::type do_simdpredict(const iterator &iter) const noexcept {
        stdx::native_simd<T> simd_vector;
        bool out_of_bound = false;
        auto prev_1 = iter.prev_addr(out_of_bound,1);
        if(prev_1 == NULL){
            simd_vector = 0;
            return simd_vector;
        }else{
            simd_vector.copy_from(prev_1, stdx::element_aligned);
            return simd_vector;
        }
    }
    
    template <uint NN = N, uint LL = L>
    inline typename std::enable_if<NN == 2 && LL == 1, stdx::native_simd<T>>::type do_simdpredict(const iterator &iter) const noexcept {
        stdx::native_simd<T> simd_vector_01;
        stdx::native_simd<T> simd_vector_10;
        stdx::native_simd<T> simd_vector_11;
        bool out_of_bound = false;
        
        T* prev_01 = iter.prev_addr(out_of_bound,0,1);
        simd_vector_01 = addr_loc(prev_01,out_of_bound);

        T* prev_10 = iter.prev_addr(out_of_bound,1,0);
        simd_vector_10 = addr_loc(prev_10,out_of_bound);

        T* prev_11 = iter.prev_addr(out_of_bound,1,1);
        simd_vector_11 = addr_loc(prev_11,out_of_bound);

        return simd_vector_01 + simd_vector_10 - simd_vector_11;
    }

    template <uint NN = N, uint LL = L>
    inline typename std::enable_if<NN == 3 && LL == 1, stdx::native_simd<T>>::type do_simdpredict(const iterator &iter) const noexcept {
        stdx::native_simd<T> simd_vector_001, simd_vector_010, simd_vector_100, simd_vector_111;
        stdx::native_simd<T> simd_vector_011, simd_vector_101, simd_vector_110;
        bool out_of_bound = false;

        T* prev_001 = iter.prev_addr(out_of_bound,0,0,1); 
        simd_vector_001 = addr_loc(prev_001,out_of_bound);

        T* prev_010 = iter.prev_addr(out_of_bound,0,1,0);
        simd_vector_010 = addr_loc(prev_010,out_of_bound);

        T* prev_100 = iter.prev_addr(out_of_bound,1,0,0);
        simd_vector_100 = addr_loc(prev_100,out_of_bound);

        T* prev_011 = iter.prev_addr(out_of_bound,0,1,1);
        simd_vector_011 = addr_loc(prev_011,out_of_bound);

        T* prev_101 = iter.prev_addr(out_of_bound,1,0,1);
        simd_vector_101 = addr_loc(prev_101,out_of_bound);

        T* prev_110 = iter.prev_addr(out_of_bound,1,1,0);
        simd_vector_110 = addr_loc(prev_110,out_of_bound);

        T* prev_111 = iter.prev_addr(out_of_bound,1,1,1);
        simd_vector_111 = addr_loc(prev_111,out_of_bound);

        return simd_vector_001 + simd_vector_010 + simd_vector_100 - simd_vector_011 - 
               simd_vector_101 - simd_vector_110 + simd_vector_111;         
    }

    template <uint NN = N, uint LL = L>
    inline typename std::enable_if<NN == 4, stdx::native_simd<T>>::type do_simdpredict(const iterator &iter) const noexcept {
        stdx::native_simd<T> simd_vector_0001, simd_vector_0010, simd_vector_0011, simd_vector_0100;
        stdx::native_simd<T> simd_vector_0101, simd_vector_0110, simd_vector_0111, simd_vector_1000;
        stdx::native_simd<T> simd_vector_1001, simd_vector_1010, simd_vector_1011, simd_vector_1100;
        stdx::native_simd<T> simd_vector_1101, simd_vector_1110, simd_vector_1111;
        bool out_of_bound = false;

        // row 1
        T* prev_0001 = iter.prev_addr(out_of_bound,0,0,0,1);
        simd_vector_0001 = addr_loc(prev_0001,out_of_bound);
        T* prev_0010 = iter.prev_addr(out_of_bound,0,0,1,0);
        simd_vector_0010 = addr_loc(prev_0010,out_of_bound);          
        T* prev_0011 = iter.prev_addr(out_of_bound,0,0,1,1);
        simd_vector_0011 = addr_loc(prev_0011,out_of_bound);
        T* prev_0100 = iter.prev_addr(out_of_bound,0,1,0,0);
        simd_vector_0100 = addr_loc(prev_0100,out_of_bound);
        
        // row 2
        T* prev_0101 = iter.prev_addr(out_of_bound,0,1,0,1);
        simd_vector_0101 = addr_loc(prev_0101,out_of_bound);
        T* prev_0110 = iter.prev_addr(out_of_bound,0,1,1,0);
        simd_vector_0110 = addr_loc(prev_0110,out_of_bound); 
        T* prev_0111 = iter.prev_addr(out_of_bound,0,1,1,1);
        simd_vector_0111 = addr_loc(prev_0111,out_of_bound);
        T* prev_1000 = iter.prev_addr(out_of_bound,1,0,0,0);
        simd_vector_1000 = addr_loc(prev_1000,out_of_bound); 

        //row 3 
        T* prev_1001 = iter.prev_addr(out_of_bound,1,0,0,1);
        simd_vector_1001 = addr_loc(prev_1001,out_of_bound);
        T* prev_1010 = iter.prev_addr(out_of_bound,1,0,1,0);
        simd_vector_1010 = addr_loc(prev_1010,out_of_bound); 
        T* prev_1011 = iter.prev_addr(out_of_bound,1,0,1,1);
        simd_vector_1011 = addr_loc(prev_1011,out_of_bound); 
        T* prev_1100 = iter.prev_addr(out_of_bound,1,1,0,0);
        simd_vector_1100 = addr_loc(prev_1100,out_of_bound);   

        //row 4
        T* prev_1101 = iter.prev_addr(out_of_bound,1,1,0,1);
        simd_vector_1101 = addr_loc(prev_1101,out_of_bound); 
        T* prev_1110 = iter.prev_addr(out_of_bound,1,1,1,0);
        simd_vector_1110 = addr_loc(prev_1110,out_of_bound); 
        T* prev_1111 = iter.prev_addr(out_of_bound,1,1,1,1);
        simd_vector_1111 = addr_loc(prev_1111,out_of_bound);

        return simd_vector_0001 + simd_vector_0010 - simd_vector_0011 + simd_vector_0100 -
               simd_vector_0101 - simd_vector_0110 + simd_vector_0111 + simd_vector_1000 -
               simd_vector_1001 - simd_vector_1010 + simd_vector_1011 - simd_vector_1100 +
               simd_vector_1101 + simd_vector_1110 - simd_vector_1111;
    }

    //Iterative prediction
    template <uint NN = N, uint LL = L>
    inline typename std::enable_if<NN == 1 && LL == 1, T>::type do_predict(const iterator &iter) const noexcept {
        return iter.prev(1);
    }

    template <uint NN = N, uint LL = L>
    inline typename std::enable_if<NN == 2 && LL == 1, T>::type do_predict(const iterator &iter) const noexcept {
        return iter.prev(0, 1) + iter.prev(1, 0) - iter.prev(1, 1);
    }

    template <uint NN = N, uint LL = L>
    inline typename std::enable_if<NN == 3 && LL == 1, T>::type do_predict(const iterator &iter) const noexcept {
        return iter.prev(0, 0, 1) + iter.prev(0, 1, 0) + iter.prev(1, 0, 0) - iter.prev(0, 1, 1) - iter.prev(1, 0, 1) -
               iter.prev(1, 1, 0) + iter.prev(1, 1, 1);
    }

    template <uint NN = N, uint LL = L>
    inline typename std::enable_if<NN == 4, T>::type do_predict(const iterator &iter) const noexcept {
        return iter.prev(0, 0, 0, 1) + iter.prev(0, 0, 1, 0) - iter.prev(0, 0, 1, 1) + iter.prev(0, 1, 0, 0) -
               iter.prev(0, 1, 0, 1) - iter.prev(0, 1, 1, 0) + iter.prev(0, 1, 1, 1) + iter.prev(1, 0, 0, 0) -
               iter.prev(1, 0, 0, 1) - iter.prev(1, 0, 1, 0) + iter.prev(1, 0, 1, 1) - iter.prev(1, 1, 0, 0) +
               iter.prev(1, 1, 0, 1) + iter.prev(1, 1, 1, 0) - iter.prev(1, 1, 1, 1);
    }
};
}  // namespace SZ3
#endif
