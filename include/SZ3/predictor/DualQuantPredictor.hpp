#ifndef SZ3_DUALQUANT_PREDICTOR_HPP
#define SZ3_DUALQUANT_PREDICTOR_HPP

#include <experimental/simd>
#include <cmath> 

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
    using block_iter = typename block_data<T, N>::block_iterator;

    DualQuantPredictor() { this->noise = 0; }

    DualQuantPredictor(double eb) {
        this->noise = 0;
        this->eb = eb;
        eb_rx2 = 1 / (2 * eb);
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
    
    bool precompress(const block_iter &) override { return true; }

    void precompress_block_commit() noexcept override {}

    bool predecompress(const block_iter &) override { return true; }

    /*
     * save stores Id along with 
     */
    void save(uchar *&c) override {
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

    ALWAYS_INLINE T estimate_error(const block_iter &block, T *d, const std::array<size_t, N> &index) override {
        //return fabs(*d - predict(block, d, index)) + this->noise;
        return 0;
    }

    ALWAYS_INLINE T predict(const block_iter &block, T *d, const std::array<size_t, N> &index) override {
        return 0;
    }

    ALWAYS_INLINE void prequant(iterator element){
        do_prequant(element);
    }
    ALWAYS_INLINE void prequant_sequential(iterator element){
        do_prequant_sequential(element);
    }
    
    // sequential lorenzo prediction
    ALWAYS_INLINE T predict(const iterator &iter) {
        return do_predict(iter);
    }

    // SIMD lorenzo prediction
    ALWAYS_INLINE stdx::native_simd<T> simd_predict(const iterator &iter) const noexcept {
        return do_simdpredict(iter);
    }
    
    // Get vector of unpred values
    ALWAYS_INLINE std::vector<T>& get_unpred_value(){
        return unpred_from_rounding_value;
    }

    // Get vector of unpred values's location
    ALWAYS_INLINE std::vector<uint64_t>& get_unpred_index(){
        return unpred_from_rounding_index;
    }

    size_t size_est() { return unpred_from_rounding_index.size() * sizeof(uint64_t) + unpred_from_rounding_value.size() * sizeof(T); }

   protected:
    T noise = 0;

   private:
    // variables
    double eb;
    double eb_rx2;

    std::vector<uint64_t> unpred_from_rounding_index;
    std::vector<T> unpred_from_rounding_value;
    

    // Prequantization process of DQ method (SIMD)
    ALWAYS_INLINE void do_prequant(iterator &iter) {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>){
            stdx::native_simd<T> orig_element;
            stdx::native_simd<T> PQ_value;

            orig_element.copy_from(&(*iter), stdx::element_aligned);
            stdx::native_simd<T> eb_reciprocal_x2 = static_cast<T>(eb_rx2);
            PQ_value = stdx::round(orig_element * eb_reciprocal_x2);

            stdx::native_simd<T> eb_x2 =  2 * static_cast<T>(eb);
            stdx::native_simd<T> reconstructed_value = PQ_value * eb_x2;
            stdx::native_simd<T> difference = orig_element - reconstructed_value;

            auto mask = stdx::fabs(difference) > static_cast<T>(eb);
            auto offset = iter.get_offset();
            const auto batch_size = orig_element.size();
            std::array<uint64_t, batch_size> local_idx;
            std::array<T, batch_size> local_val;
            std::size_t count = 0;
            for(std::size_t i = 0; i < mask.size(); ++i){
                if(mask[i]){
                    local_idx[count] = offset + i;
                    local_val[count] = orig_element[i];
                    ++count;
                }      
            }
            unpred_from_rounding_index.insert(unpred_from_rounding_index.end(),
                                  local_idx.begin(), local_idx.begin() + count);
            unpred_from_rounding_value.insert(unpred_from_rounding_value.end(),
                                  local_val.begin(), local_val.begin() + count);
            PQ_value.copy_to(&(*iter), stdx::element_aligned);
        }else{
            stdx::native_simd<T> orig_element;
            stdx::native_simd<T> PQ_value;

            orig_element.copy_from(&(*iter), stdx::element_aligned);
            stdx::native_simd<T> eb_reciprocal_x2 = static_cast<T>(eb_rx2);
            PQ_value = orig_element * eb_reciprocal_x2;

            stdx::native_simd<T> eb_x2 =  2 * static_cast<T>(eb);
            stdx::native_simd<T> reconstructed_value = PQ_value * eb_x2;
            stdx::native_simd<T> difference = orig_element - reconstructed_value;

            auto mask = stdx::abs(difference) > static_cast<T>(eb);
            auto offset = iter.get_offset();
            const auto batch_size = orig_element.size();
            std::array<uint64_t, batch_size> local_idx;
            std::array<T, batch_size> local_val;
            std::size_t count = 0;
            for(std::size_t i = 0; i < mask.size(); ++i){
                if(mask[i]){
                    local_idx[count] = offset + i;
                    local_val[count] = orig_element[i];
                    ++count;
                }      
            }
            unpred_from_rounding_index.insert(unpred_from_rounding_index.end(),
                                  local_idx.begin(), local_idx.begin() + count);
            unpred_from_rounding_value.insert(unpred_from_rounding_value.end(),
                                  local_val.begin(), local_val.begin() + count);
            PQ_value.copy_to(&(*iter), stdx::element_aligned);
        }
    }

    // Prequantization process of DQ method (sequential cases)
    ALWAYS_INLINE void do_prequant_sequential(iterator &iter) {
        auto PQ_value = *iter * eb_rx2;
        auto reconstructed_value = PQ_value * 2 *eb;
        auto difference = *iter - reconstructed_value;
        if(std::fabs(difference) > eb){
            unpred_from_rounding_index.push_back(iter.get_offset());
            unpred_from_rounding_value.push_back(*iter);
        }
        *iter = std::round(PQ_value);
    }

    // return SIMD vector based on the prev_address location values
    ALWAYS_INLINE stdx::native_simd<T> addr_location(T* address, bool &out_of_bound) const {
        stdx::native_simd<T> temp_vector = 0;
        if(address != NULL){
            temp_vector.copy_from(address, stdx::element_aligned);
            if(out_of_bound){
                temp_vector[0] = 0;
                out_of_bound = false;
            }
        }
        return temp_vector;

    }

    template <uint NN = N, uint LL = L>
    ALWAYS_INLINE typename std::enable_if<NN == 1 && LL == 1, stdx::native_simd<T>>::type do_simdpredict(const iterator &iter) const noexcept {
        bool out_of_bound = false;
        T* prev_1 = iter.prev_address(out_of_bound,1);
        return addr_location(prev_1,out_of_bound);
    }
    
    template <uint NN = N, uint LL = L>
    ALWAYS_INLINE typename std::enable_if<NN == 2 && LL == 1, stdx::native_simd<T>>::type do_simdpredict(const iterator &iter) const noexcept {
        stdx::native_simd<T> simd_vector_01;
        stdx::native_simd<T> simd_vector_10;
        stdx::native_simd<T> simd_vector_11;
        bool out_of_bound = false;
        
        T* prev_01 = iter.prev_address(out_of_bound,0,1);
        simd_vector_01 = addr_location(prev_01,out_of_bound);

        T* prev_10 = iter.prev_address(out_of_bound,1,0);
        simd_vector_10 = addr_location(prev_10,out_of_bound);

        T* prev_11 = iter.prev_address(out_of_bound,1,1);
        simd_vector_11 = addr_location(prev_11,out_of_bound);

        return simd_vector_01 + simd_vector_10 - simd_vector_11;
    }

    template <uint NN = N, uint LL = L>
    ALWAYS_INLINE typename std::enable_if<NN == 3 && LL == 1, stdx::native_simd<T>>::type do_simdpredict(const iterator &iter) const noexcept {
        stdx::native_simd<T> simd_vector_001, simd_vector_010, simd_vector_100, simd_vector_111;
        stdx::native_simd<T> simd_vector_011, simd_vector_101, simd_vector_110;
        bool out_of_bound = false;

        T* prev_001 = iter.prev_address(out_of_bound,0,0,1); 
        simd_vector_001 = addr_location(prev_001,out_of_bound);

        T* prev_010 = iter.prev_address(out_of_bound,0,1,0);
        simd_vector_010 = addr_location(prev_010,out_of_bound);

        T* prev_100 = iter.prev_address(out_of_bound,1,0,0);
        simd_vector_100 = addr_location(prev_100,out_of_bound);

        T* prev_011 = iter.prev_address(out_of_bound,0,1,1);
        simd_vector_011 = addr_location(prev_011,out_of_bound);

        T* prev_101 = iter.prev_address(out_of_bound,1,0,1);
        simd_vector_101 = addr_location(prev_101,out_of_bound);

        T* prev_110 = iter.prev_address(out_of_bound,1,1,0);
        simd_vector_110 = addr_location(prev_110,out_of_bound);

        T* prev_111 = iter.prev_address(out_of_bound,1,1,1);
        simd_vector_111 = addr_location(prev_111,out_of_bound);

        return simd_vector_001 + simd_vector_010 + simd_vector_100 - simd_vector_011 - 
               simd_vector_101 - simd_vector_110 + simd_vector_111;         
    }

    template <uint NN = N, uint LL = L>
    ALWAYS_INLINE typename std::enable_if<NN == 4, stdx::native_simd<T>>::type do_simdpredict(const iterator &iter) const noexcept {
        stdx::native_simd<T> simd_vector_0001, simd_vector_0010, simd_vector_0011, simd_vector_0100;
        stdx::native_simd<T> simd_vector_0101, simd_vector_0110, simd_vector_0111, simd_vector_1000;
        stdx::native_simd<T> simd_vector_1001, simd_vector_1010, simd_vector_1011, simd_vector_1100;
        stdx::native_simd<T> simd_vector_1101, simd_vector_1110, simd_vector_1111;
        bool out_of_bound = false;

        // row 1
        T* prev_0001 = iter.prev_address(out_of_bound,0,0,0,1);
        simd_vector_0001 = addr_location(prev_0001,out_of_bound);
        T* prev_0010 = iter.prev_address(out_of_bound,0,0,1,0);
        simd_vector_0010 = addr_location(prev_0010,out_of_bound);          
        T* prev_0011 = iter.prev_address(out_of_bound,0,0,1,1);
        simd_vector_0011 = addr_location(prev_0011,out_of_bound);
        T* prev_0100 = iter.prev_address(out_of_bound,0,1,0,0);
        simd_vector_0100 = addr_location(prev_0100,out_of_bound);
        
        // row 2
        T* prev_0101 = iter.prev_address(out_of_bound,0,1,0,1);
        simd_vector_0101 = addr_location(prev_0101,out_of_bound);
        T* prev_0110 = iter.prev_address(out_of_bound,0,1,1,0);
        simd_vector_0110 = addr_location(prev_0110,out_of_bound); 
        T* prev_0111 = iter.prev_address(out_of_bound,0,1,1,1);
        simd_vector_0111 = addr_location(prev_0111,out_of_bound);
        T* prev_1000 = iter.prev_address(out_of_bound,1,0,0,0);
        simd_vector_1000 = addr_location(prev_1000,out_of_bound); 

        //row 3 
        T* prev_1001 = iter.prev_address(out_of_bound,1,0,0,1);
        simd_vector_1001 = addr_location(prev_1001,out_of_bound);
        T* prev_1010 = iter.prev_address(out_of_bound,1,0,1,0);
        simd_vector_1010 = addr_location(prev_1010,out_of_bound); 
        T* prev_1011 = iter.prev_address(out_of_bound,1,0,1,1);
        simd_vector_1011 = addr_location(prev_1011,out_of_bound); 
        T* prev_1100 = iter.prev_address(out_of_bound,1,1,0,0);
        simd_vector_1100 = addr_location(prev_1100,out_of_bound);   

        //row 4
        T* prev_1101 = iter.prev_address(out_of_bound,1,1,0,1);
        simd_vector_1101 = addr_location(prev_1101,out_of_bound); 
        T* prev_1110 = iter.prev_address(out_of_bound,1,1,1,0);
        simd_vector_1110 = addr_location(prev_1110,out_of_bound); 
        T* prev_1111 = iter.prev_address(out_of_bound,1,1,1,1);
        simd_vector_1111 = addr_location(prev_1111,out_of_bound);

        return simd_vector_0001 + simd_vector_0010 - simd_vector_0011 + simd_vector_0100 -
               simd_vector_0101 - simd_vector_0110 + simd_vector_0111 + simd_vector_1000 -
               simd_vector_1001 - simd_vector_1010 + simd_vector_1011 - simd_vector_1100 +
               simd_vector_1101 + simd_vector_1110 - simd_vector_1111;
    }

    //Iterative prediction
    template <uint NN = N, uint LL = L>
    ALWAYS_INLINE typename std::enable_if<NN == 1 && LL == 1, T>::type do_predict(const iterator &iter) const noexcept {
        return iter.prev(1);
    }

    template <uint NN = N, uint LL = L>
    ALWAYS_INLINE typename std::enable_if<NN == 2 && LL == 1, T>::type do_predict(const iterator &iter) const noexcept {
        return iter.prev(0, 1) + iter.prev(1, 0) - iter.prev(1, 1);
    }

    template <uint NN = N, uint LL = L>
    ALWAYS_INLINE typename std::enable_if<NN == 3 && LL == 1, T>::type do_predict(const iterator &iter) const noexcept {
        return iter.prev(0, 0, 1) + iter.prev(0, 1, 0) + iter.prev(1, 0, 0) - iter.prev(0, 1, 1) - iter.prev(1, 0, 1) -
               iter.prev(1, 1, 0) + iter.prev(1, 1, 1);
    }

    template <uint NN = N, uint LL = L>
    ALWAYS_INLINE typename std::enable_if<NN == 4, T>::type do_predict(const iterator &iter) const noexcept {
        return iter.prev(0, 0, 0, 1) + iter.prev(0, 0, 1, 0) - iter.prev(0, 0, 1, 1) + iter.prev(0, 1, 0, 0) -
               iter.prev(0, 1, 0, 1) - iter.prev(0, 1, 1, 0) + iter.prev(0, 1, 1, 1) + iter.prev(1, 0, 0, 0) -
               iter.prev(1, 0, 0, 1) - iter.prev(1, 0, 1, 0) + iter.prev(1, 0, 1, 1) - iter.prev(1, 1, 0, 0) +
               iter.prev(1, 1, 0, 1) + iter.prev(1, 1, 1, 0) - iter.prev(1, 1, 1, 1);
    }
};
}  // namespace SZ3
#endif