#ifndef _SZ_DUALQUANT_HPP
#define _SZ_DUALQUANT_HPP

#include <cassert>
#include <experimental/simd>

#include "SZ3/def.hpp"
#include "SZ3/predictor/Predictor.hpp"
#include "SZ3/utils/Iterator.hpp"

// get sqyuentail working

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
            simd_prequant(element);
        }
        return true;
    }

    void precompress_block_commit() noexcept override {}

    bool predecompress_block(const std::shared_ptr<Range> &) override { return true; }

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
        return fabs(*iter - predict(iter)) + this->noise;
    }

    T simd_prequant(iterator &iter) noexcept override {
        //   printf("the iterator is %f\n", iter);
        // printsimd(simd_vector);
        prequant(iter);
        // printf("the iterator after is %f\n", iter);
        // return do_predict(iter);
        //  do_predict(iter);
        return 0;
    }

    // T simdpredict2(const iterator &iter) const noexcept override {
    //     // printf("the iterator is %f\n", iter);
    //     return do_predict(iter);
    // }

    T predict(const iterator &iter) const noexcept override {
        // printf("the iterator is %f\n", iter);
        // stdx::native_simd<float> simd_vector;
        // simd_vector.copy_from(&(*iter), stdx::element_aligned);
        // // printsimd(simd_vector);
        // prequant(simd_vector, iter);
        // printf("the iterator after is %f\n", iter);
        // return simd_predict(iter);
        return 0;
    }

    T simd_predict(iterator &iter) noexcept override {
        // printf("the iterator is %f\n", iter);
        // stdx::native_simd<float> simd_vector;
        // simd_vector.copy_from(&(*iter), stdx::element_aligned);
        // // printsimd(simd_vector);
        // prequant(simd_vector, iter);
        // printf("the iterator after is %f\n", iter);
        return do_predict(iter);
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
    // void prequant(const stdx::native_simd<float> &simd_vector, const iterator &iter) {

    void prequant(iterator &iter) {
        // printf("the error is %f", eb);  // Perform operations using simd_vector
        stdx::native_simd<float> simd_vector;
        simd_vector.copy_from(&(*iter), stdx::element_aligned);
        float divisor = static_cast<float>(2 * eb);
        // Perform element-wise division
        stdx::native_simd<float> temp_vector = simd_vector / divisor;
        // float dataw[4];
        temp_vector.copy_to(&(*iter), stdx::element_aligned);
        printf("prequantizing\n");
        printsimd(temp_vector);
    }
    // void prequant2(const stdx::native_simd<float> &simd_vector, &element) {
    //     printf("the error is %f", eb);  // Perform operations using simd_vector
    //     float divisor = static_cast<float>(2 * eb);
    //     // Perform element-wise division
    //     stdx::native_simd<float> temp_vector = simd_vector / divisor;
    //     //  printf("the copy index is %d:\n", copy_index);
    //     temp_vector.copy_to(&element, stdx::vector_aligned);
    //     printsimd(temp_vector);
    // }
    template <uint NN = N, uint LL = L>
    // Added functions

    inline typename std::enable_if<NN == 1 && LL == 1, T>::type do_predict(iterator &iter) const noexcept {
        // printf("here 1\n");
        // using simd_t = stdx::native_simd<float>;
        // stdx::native_simd<float> simd_vector;
        // simd_vector.copy_from(&(*iter), stdx::element_aligned);
        // iter.prev(1);
        // printf("the iter prev value is %f: \n", iter.prev(1));
        // for (auto i = 0; i < 4; ++i) {
        //     iter.prev(1)
        // const float *ptr_prev01 = &iter.prev(1);
        // simd_t prev01 = simd_t::copy_from(ptr_prev01, stdx::element_aligned);
        // printf("in 1 here %f\n", ptr_prev01);
        // printsimd(prev01);
        //  }
        // std::cout << "Iter is" << iter << std::endl;
        // ++iter;
        // constexpr int simd_size = 4;
        // alignas(64) float temp_prev01[simd_size];
        // for (int i = 0; i < simd_size; ++i) {
        //     temp_prev01[i] = ().prev(1);
        //     // temp_prev10[i] = (iter + i).prev(1, 0);
        //     // temp_prev11[i] = (iter + i).prev(1, 1);
        // }
        // stdx::native_simd<float> prev01 = stdx::native_simd<float>::copy_from(temp_prev01, stdx::element_aligned);
        // printsimd(prev01);

        // Bad idea: doing this because const and have to modify a lot of places but this incurs copy cost(or not?)
        // iterator temp_iter = iter;
        // //++temp_iter;
        // constexpr int simd_size = 4;
        // alignas(64) float temp_prev01[simd_size];
        // using simd_t = stdx::native_simd<float>;
        // for (int i = 0; i < simd_size; ++i) {
        //     temp_prev01[i] = (++temp_iter).prev(1);
        //     // temp_prev10[i] = (iter + i).prev(1, 0);
        //     // temp_prev11[i] = (iter + i).prev(1, 1);
        // }
        // stdx::native_simd<float> prev01;
        // prev01.copy_from(&(*temp_prev01), stdx::element_aligned);
        // // stdx::native_simd<float> result = prev01;
        // printsimd(prev01);
        // return prev01;
        // const double *current_address = &(*iter);

        // stdx::native_simd<float> simd_vector;
        auto *simd_vector = new stdx::native_simd<float>;

        simd_vector->copy_from(&(*(--iter)), stdx::element_aligned);
        // printsimd(simd_vector);
        return simd_vector;
    }

    template <uint NN = N, uint LL = L>
    inline typename std::enable_if<NN == 2 && LL == 1, T>::type do_predict(const iterator &iter) const noexcept {
        // New data pass here
        // Same operations on all 4 data
        // printf("here 2\n");

        // using simd_t = stdx::native_simd<float>;      // or use `stdx::fixed_size_simd<float, 16>` for a fixed size
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
        // printsimd(result);

        return iter.prev(0, 1) + iter.prev(1, 0) - iter.prev(1, 1);
    }

    template <uint NN = N, uint LL = L>
    inline typename std::enable_if<NN == 3 && LL == 1, T>::type do_predict(const iterator &iter) const noexcept {
        printf("here 3\n");
        return iter.prev(0, 0, 1) + iter.prev(0, 1, 0) + iter.prev(1, 0, 0) - iter.prev(0, 1, 1) - iter.prev(1, 0, 1) -
               iter.prev(1, 1, 0) + iter.prev(1, 1, 1);
    }

    template <uint NN = N, uint LL = L>
    inline typename std::enable_if<NN == 4, T>::type do_predict(const iterator &iter) const noexcept {
        printf("here 4\n");
        return iter.prev(0, 0, 0, 1) + iter.prev(0, 0, 1, 0) - iter.prev(0, 0, 1, 1) + iter.prev(0, 1, 0, 0) -
               iter.prev(0, 1, 0, 1) - iter.prev(0, 1, 1, 0) + iter.prev(0, 1, 1, 1) + iter.prev(1, 0, 0, 0) -
               iter.prev(1, 0, 0, 1) - iter.prev(1, 0, 1, 0) + iter.prev(1, 0, 1, 1) - iter.prev(1, 1, 0, 0) +
               iter.prev(1, 1, 0, 1) + iter.prev(1, 1, 1, 0) - iter.prev(1, 1, 1, 1);
    }

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
};
}  // namespace SZ3
#endif
