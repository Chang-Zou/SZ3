#ifndef _SZ_DUALQUANT_HPP
#define _SZ_DUALQUANT_HPP

#include <cassert>
#include <experimental/simd>
//#include </home/changfz/.local/include/vir/simd.h> // direct path need change
#include <cmath> 

#include "SZ3/def.hpp"
#include "SZ3/predictor/Predictor.hpp"
#include "SZ3/utils/Iterator.hpp"

// get sqyuentail working
//namespace stdx = vir::stdx; //// works require c++20

namespace stdx {
    using namespace std::experimental;
    using namespace std::experimental::__proposed;
}

// get the size
//template< class T, class Abi = stdx::simd_abi::compatible<T>>
//constexpr std::size_t simd_size_v = stdx::simd_size<T, Abi>::value;
// radius of vecsz is always 2048
namespace SZ3 {

// N-dimension L-layer dualquant predictor
template <class T, uint N, uint L>
class DualQuantPredictor : public concepts::PredictorInterface<T, N> {
   public:
    static const uint8_t predictor_id = 0b00000100;
    using Range = multi_dimensional_range<T, N>;
    using iterator = typename multi_dimensional_range<T, N>::iterator;

    DualQuantPredictor() { this->noise = 0; }

    // This is where the error bound value is being passed in when it is constructed?
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
        size_t batch_size = 4;
        for (auto element = element_range->begin(); element != element_range->end(); element += batch_size) {
            //printf("in element prequant value %f \n", *element);
            prequant<float>(element);
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
        //return 0;
        return do_predict(iter);
    }

    
    inline stdx::native_simd<T> simd_predict(const iterator &iter) const noexcept {
        // printf("prediction \n");
        // printsimd(do_predict(iter));
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
        // printf("the error is %f", eb);  // Perform operations using simd_vector
        stdx::native_simd<TT> simd_vector;
        simd_vector.copy_from(&(*iter), stdx::element_aligned);
        stdx::native_simd<TT> multipler = static_cast<TT>(ebs_L4);
        // Perform element-wise division
        stdx::native_simd<TT> temp_vector = stdx::round(simd_vector * multipler);
        // float dataw[4];
        temp_vector.copy_to(&(*iter), stdx::element_aligned);
        // printf("prequantizing\n");
        // printsimd(temp_vector);
    }

    template <uint NN = N, uint LL = L>
    inline typename std::enable_if<NN == 1 && LL == 1, stdx::native_simd<T>>::type do_simdpredict(const iterator &iter) const noexcept {
        // printf("iter in here is:%f\n", *iter);
        stdx::native_simd<T> simd_vector;
        iterator temp_iter = iter;
        simd_vector.copy_from(&(*--temp_iter), stdx::element_aligned);
        //printsimd(simd_vector);
        //++iter;
        
        
        return simd_vector;
    }
    
    template <uint NN = N, uint LL = L>
    inline typename std::enable_if<NN == 2 && LL == 1, T>::type do_simdpredict(const iterator &iter) const noexcept {
        // New data pass here
        // Same operations on all 4 data
        // printf("here 2\n");

        // using simd_t = stdx::native_simd<T>;      // or use `stdx::fixed_size_simd<float, 16>` for a fixed size
        // constexpr size_t simd_size = simd_t::size();  // Number of elements processed in parallel

        // // Prepare data
        // const float *ptr_prev01 = &element.prev(0, 1);  // Pointer to prev(0, 1) elements
        // const float *ptr_prev10 = &element.prev(1, 0);  // Pointer to prev(1, 0) elements
        // const float *ptr_prev11 = &element.prev(1, 1);  // Pointer to prev(1, 1) elements

        // // Load data into SIMD registers
        // simd_t prev01 = simd_t::copy_from(ptr_prev01, stdx::element_aligned);
        // simd_t prev10 = simd_t::copy_from(ptr_prev10, stdx::element_aligned);
        // simd_t prev11 = simd_t::copy_from(ptr_prev11, stdx::element_aligned);

        // // Perform SIMD operations
        // simd_t result = prev01 + prev10 - prev11;

        // // Debug print (optional)

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


    /*
    template <uint NN = N, uint LL = L>
    inline typename std::enable_if<NN == 1 && LL == 2, T>::type do_predict(const iterator &iter) const noexcept {
        printf("here 5\n");
        return 2 * iter.prev(1) - iter.prev(2);
    }

    template <uint NN = N, uint LL = L>
    inline typename std::enable_if<NN == 2 && LL == 2, T>::type do_predict(const iterator &iter) const noexcept {
        printf("here 6\n");
        return 2 * iter.prev(0, 1) - iter.prev(0, 2) + 2 * iter.prev(1, 0) - 4 * iter.prev(1, 1) + 2 * iter.prev(1, 2) -
               iter.prev(2, 0) + 2 * iter.prev(2, 1) - iter.prev(2, 2);
    }

    template <uint NN = N, uint LL = L>
    inline typename std::enable_if<NN == 3 && LL == 2, T>::type do_predict(const iterator &iter) const noexcept {
        printf("here 7\n");
        return 2 * iter.prev(0, 0, 1) - iter.prev(0, 0, 2) + 2 * iter.prev(0, 1, 0) - 4 * iter.prev(0, 1, 1) +
               2 * iter.prev(0, 1, 2) - iter.prev(0, 2, 0) + 2 * iter.prev(0, 2, 1) - iter.prev(0, 2, 2) +
               2 * iter.prev(1, 0, 0) - 4 * iter.prev(1, 0, 1) + 2 * iter.prev(1, 0, 2) - 4 * iter.prev(1, 1, 0) +
               8 * iter.prev(1, 1, 1) - 4 * iter.prev(1, 1, 2) + 2 * iter.prev(1, 2, 0) - 4 * iter.prev(1, 2, 1) +
               2 * iter.prev(1, 2, 2) - iter.prev(2, 0, 0) + 2 * iter.prev(2, 0, 1) - iter.prev(2, 0, 2) +
               2 * iter.prev(2, 1, 0) - 4 * iter.prev(2, 1, 1) + 2 * iter.prev(2, 1, 2) - iter.prev(2, 2, 0) +
               2 * iter.prev(2, 2, 1) - iter.prev(2, 2, 2);
    }
    */
};
}  // namespace SZ3
#endif
