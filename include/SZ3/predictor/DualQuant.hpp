#ifndef _SZ_DUALQUANT_HPP
#define _SZ_DUALQUANT_HPP

#include <cassert>
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

    DualQuantPredictor() { this->noise = 0; }

    DualQuantPredictor(double eb) {
        this->noise = 0;
        this->eb = eb;
        ebs_L4 = 1 / (2 * eb);
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
        size_t batch_size = stdx::native_simd<T>::size();
        size_t element_size = element_range->get_dimensions()[0];
        auto element = element_range->begin();

        if(element_size % batch_size == 0){
            for(; element != element_range->end(); element += batch_size) {
                prequant<float>(element);
            }
        }else{
            size_t element_end = 0;
            switch(batch_size){
                case 16:
                    element_end = (element_size & ~0xF);
                    break;
                case 8:
                    element_end = (element_size & ~0x7);
                    break;
                case 4:
                    element_end = (element_size & ~0x3);
                    break;
                default:
                    element_end = (element_size & ~0x3);
                    break;   
            }
            for(; element.get_local_index(0) < element_end; element += batch_size) { prequant<float>(element);}
            for(; element != element_range->end(); element++) { prequant_sequential<float>(element);}
        }
        return true;
    }

    void precompress_block_commit() noexcept override {}

    bool predecompress_block(const std::shared_ptr<Range> &) override {return true;}

    /*
     * save doesn't need to store anything except the id
     */
    // std::string save() const {
    //   return std::string(1, predictor_id);
    // }
    void save(uchar *&c) const override {
        c[0] = predictor_id;
        c += sizeof(uint8_t);
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
    
    //        void clear() {}

   protected:
    T noise = 0;

   private:
    // variables
    double eb;
    double ebs_L4;
    void printsimd(auto const &a) const {
        for (std::size_t i{}; i != std::size(a); ++i) std::cout << a[i] << ' ';
        std::cout << '\n';
    }
  
    template<class TT>
    inline void prequant(iterator &iter) {
        stdx::native_simd<TT> simd_vector;
        simd_vector.copy_from(&(*iter), stdx::element_aligned);
        stdx::native_simd<TT> multipler = static_cast<TT>(ebs_L4);
        stdx::native_simd<TT> temp_vector = stdx::round(simd_vector * multipler);
        temp_vector.copy_to(&(*iter), stdx::element_aligned);
    }

    template<class TT>
    inline void prequant_sequential(iterator &iter) {
        *iter = std::round(*iter * ebs_L4);
    }


    template <uint NN = N, uint LL = L>
    inline typename std::enable_if<NN == 1 && LL == 1, stdx::native_simd<T>>::type do_simdpredict(const iterator &iter) const noexcept {
        stdx::native_simd<T> simd_vector;
        iterator temp_iter = iter;
        simd_vector.copy_from(&(*--temp_iter), stdx::element_aligned);
        return simd_vector;
    }
    
    template <uint NN = N, uint LL = L>
    inline typename std::enable_if<NN == 2 && LL == 1, T>::type do_simdpredict(const iterator &iter) const noexcept {

        return iter.prev(0, 1) + iter.prev(1, 0) - iter.prev(1, 1);
    }

    template <uint NN = N, uint LL = L>
    inline typename std::enable_if<NN == 3 && LL == 1, T>::type do_simdpredict(const iterator &iter) const noexcept {
        printf("here 3\n");
        return iter.prev(0, 0, 1) + iter.prev(0, 1, 0) + iter.prev(1, 0, 0) - iter.prev(0, 1, 1) - iter.prev(1, 0, 1) -
               iter.prev(1, 1, 0) + iter.prev(1, 1, 1);
    }

    template <uint NN = N, uint LL = L>
    inline typename std::enable_if<NN == 4, T>::type do_simdpredict(const iterator &iter) const noexcept {
        printf("here 4\n");
        return iter.prev(0, 0, 0, 1) + iter.prev(0, 0, 1, 0) - iter.prev(0, 0, 1, 1) + iter.prev(0, 1, 0, 0) -
               iter.prev(0, 1, 0, 1) - iter.prev(0, 1, 1, 0) + iter.prev(0, 1, 1, 1) + iter.prev(1, 0, 0, 0) -
               iter.prev(1, 0, 0, 1) - iter.prev(1, 0, 1, 0) + iter.prev(1, 0, 1, 1) - iter.prev(1, 1, 0, 0) +
               iter.prev(1, 1, 0, 1) + iter.prev(1, 1, 1, 0) - iter.prev(1, 1, 1, 1);
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

    template <uint NN = N, uint LL = L>
    inline typename std::enable_if<NN == 1 && LL == 2, T>::type do_predict(const iterator &iter) const noexcept {
        return 2 * iter.prev(1) - iter.prev(2);
    }

    template <uint NN = N, uint LL = L>
    inline typename std::enable_if<NN == 2 && LL == 2, T>::type do_predict(const iterator &iter) const noexcept {
        return 2 * iter.prev(0, 1) - iter.prev(0, 2) + 2 * iter.prev(1, 0) - 4 * iter.prev(1, 1) + 2 * iter.prev(1, 2) -
               iter.prev(2, 0) + 2 * iter.prev(2, 1) - iter.prev(2, 2);
    }

    template <uint NN = N, uint LL = L>
    inline typename std::enable_if<NN == 3 && LL == 2, T>::type do_predict(const iterator &iter) const noexcept {
        return 2 * iter.prev(0, 0, 1) - iter.prev(0, 0, 2) + 2 * iter.prev(0, 1, 0) - 4 * iter.prev(0, 1, 1) +
               2 * iter.prev(0, 1, 2) - iter.prev(0, 2, 0) + 2 * iter.prev(0, 2, 1) - iter.prev(0, 2, 2) +
               2 * iter.prev(1, 0, 0) - 4 * iter.prev(1, 0, 1) + 2 * iter.prev(1, 0, 2) - 4 * iter.prev(1, 1, 0) +
               8 * iter.prev(1, 1, 1) - 4 * iter.prev(1, 1, 2) + 2 * iter.prev(1, 2, 0) - 4 * iter.prev(1, 2, 1) +
               2 * iter.prev(1, 2, 2) - iter.prev(2, 0, 0) + 2 * iter.prev(2, 0, 1) - iter.prev(2, 0, 2) +
               2 * iter.prev(2, 1, 0) - 4 * iter.prev(2, 1, 1) + 2 * iter.prev(2, 1, 2) - iter.prev(2, 2, 0) +
               2 * iter.prev(2, 2, 1) - iter.prev(2, 2, 2);
    }
};
}  // namespace SZ3
#endif
