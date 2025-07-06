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
        if(N == 1){
            size_t batch_size = stdx::native_simd<T>::size();
            size_t element_size = element_range->end().get_offset() - element_range->begin().get_offset();
            auto element = element_range->begin();
            auto element2 = element.get_dimensions();

            if(element_size % batch_size == 0){
                for(; element != element_range->end(); element += batch_size) {
                    prequant<float>(element);
                }
            }else{
                size_t element_end = 0;
                size_t count = 0;
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
                for(; count < element_end; element += batch_size) { 
                    prequant<float>(element);
                    count +=batch_size;
                }
                for(; element != element_range->end(); ++element) {
                    prequant_sequential<float>(element);
                }
            }
            return true;
        }else if (N == 2){ // N == 2
            size_t batch_size = stdx::native_simd<T>::size();
            size_t element_size = element_range->end().get_offset() - element_range->begin().get_offset();
            auto element = element_range->begin();
            auto row = element.get_dimensions()[0];
            auto col = element.get_dimensions()[1];

            if((element_size % batch_size == 0) && (col % batch_size == 0)){ // case 1 & 2
                for(; element != element_range->end(); element += batch_size) {
                    prequant<float>(element);
                }
            }else if (col > batch_size && col % batch_size != 0){ // case 3
                while(element != element_range->end()){
                    size_t count = 0;
                    for(; count + batch_size < col; element+=batch_size){
                        prequant<float>(element);
                        count +=batch_size;
                    }
                    for(; count < col; ++element){
                        prequant_sequential<float>(element);
                        count++;
                    }
                }
            }else{
                for(; element != element_range->end(); ++element) {prequant_sequential<float>(element);}
            }
            return true;
        }else{ // N == 3
            size_t batch_size = stdx::native_simd<T>::size();
            size_t element_size = element_range->end().get_offset() - element_range->begin().get_offset();
            auto element = element_range->begin();
            auto depth = element.get_dimensions()[0];
            auto row = element.get_dimensions()[1];
            auto col = element.get_dimensions()[2];

            if((element_size % batch_size == 0) && (col % batch_size == 0)){ // case 1 & 2
                for(; element != element_range->end(); element += batch_size) {
                    prequant<float>(element);
                }
            }else if (col > batch_size && col % batch_size != 0){ // case 3
                while(element != element_range->end()){
                    size_t count = 0;
                    for(; count + batch_size < col; element+=batch_size){
                        prequant<float>(element);
                        count +=batch_size;
                    }
                    for(; count < col; ++element){
                        prequant_sequential<float>(element);
                        count++;
                    }
                }
            }else{
                for(; element != element_range->end(); ++element) {prequant_sequential<float>(element);}
            }
            return true;
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
        int* out_of_bound = new int;
        *out_of_bound = 0;
        auto prev_1 = iter.prevaddr(out_of_bound,1);
        if(prev_1 == NULL){
            simd_vector = 0;
        }else{
            simd_vector.copy_from(prev_1, stdx::element_aligned);
        }
        return simd_vector;
    }
    
    inline stdx::native_simd<T> addrloc(T* address, int *out_of_bound) const {
        stdx::native_simd<T> temp_vector;
        if(address == NULL){
            temp_vector = 0;
        }else if(* out_of_bound == 1){
            temp_vector.copy_from(address, stdx::element_aligned);
            temp_vector[0] = 0;
            *out_of_bound = 0;
        }else{
            temp_vector.copy_from(address, stdx::element_aligned);
        }
        return temp_vector;
    }

    template <uint NN = N, uint LL = L>
    inline typename std::enable_if<NN == 2 && LL == 1, stdx::native_simd<T>>::type do_simdpredict(const iterator &iter) const noexcept {
        stdx::native_simd<T> simd_vector_01;
        stdx::native_simd<T> simd_vector_10;
        stdx::native_simd<T> simd_vector_11;
        int* out_of_bound = new int;
        *out_of_bound = 0;
        
        T* prev_01 = iter.prevaddr(out_of_bound,0,1);
        simd_vector_01 = addrloc(prev_01,out_of_bound);

        T* prev_10 = iter.prevaddr(out_of_bound,1,0);
        simd_vector_10 = addrloc(prev_10,out_of_bound);

        T* prev_11 = iter.prevaddr(out_of_bound,1,1);
        simd_vector_11 = addrloc(prev_11,out_of_bound);
        // auto prev_01 = iter.prevaddr(out_of_bound,0,1);
        // if(prev_01 == NULL){
        //     simd_vector_01 = 0;
        // }else if(*out_of_bound == 1){
        //     simd_vector_01.copy_from(prev_01, stdx::element_aligned);
        //     simd_vector_01[0] = 0;
        //     *out_of_bound = 0;
        // }else{
        //     simd_vector_01.copy_from(prev_01, stdx::element_aligned);
        // }

        // auto prev_10 = iter.prevaddr(out_of_bound,1,0);
        // if(prev_10 == NULL){
        //     simd_vector_10 = 0;
        // }else if (*out_of_bound == 1){
        //     simd_vector_10.copy_from(prev_10, stdx::element_aligned);
        //     simd_vector_10[0] = 0;
        //     *out_of_bound = 0;
        // }else{
        //     simd_vector_10.copy_from(prev_10, stdx::element_aligned);
        // }

        // auto prev_11 = iter.prevaddr(out_of_bound,1,1);
        // if(prev_11 == NULL){
        //     simd_vector_11 = 0;
        // }else if (*out_of_bound == 1){
        //     simd_vector_11.copy_from(prev_11, stdx::element_aligned);
        //     simd_vector_11[0] = 0;
        //     *out_of_bound = 0;
        // }else{
        //     simd_vector_11.copy_from(prev_11, stdx::element_aligned);
        // }
        return simd_vector_01 + simd_vector_10 - simd_vector_11;
    }

    template <uint NN = N, uint LL = L>
    inline typename std::enable_if<NN == 3 && LL == 1, T>::type do_simdpredict(const iterator &iter) const noexcept {
        stdx::native_simd<T> simd_vector_001;
        stdx::native_simd<T> simd_vector_010;
        stdx::native_simd<T> simd_vector_100;
        stdx::native_simd<T> simd_vector_011;
        stdx::native_simd<T> simd_vector_101;
        stdx::native_simd<T> simd_vector_110;
        stdx::native_simd<T> simd_vector_111;
        int* out_of_bound = new int;
        *out_of_bound = 0;

        auto prev_001 = iter.prevaddr(out_of_bound,0,0,1);
        if(prev_001 == NULL){
            simd_vector_001 = 0;
        }else if(*out_of_bound == 1){
            simd_vector_001.copy_from(prev_001, stdx::element_aligned);
            simd_vector_001[0] = 0;
            *out_of_bound = 0;
        }else{
            simd_vector_001.copy_from(prev_001, stdx::element_aligned);
        }

        auto prev_010 = iter.prevaddr(out_of_bound,0,1,0);
        if(prev_010 == NULL){
            simd_vector_010 = 0;
        }else if(*out_of_bound == 1){
            simd_vector_010.copy_from(prev_010, stdx::element_aligned);
            simd_vector_010[0] = 0;
            *out_of_bound = 0;
        }else{
            simd_vector_010.copy_from(prev_010, stdx::element_aligned);
        }

        auto prev_100 = iter.prevaddr(out_of_bound,1,0,0);
        if(prev_100 == NULL){
            simd_vector_100 = 0;
        }else if(*out_of_bound == 1){
            simd_vector_100.copy_from(prev_100, stdx::element_aligned);
            simd_vector_100[0] = 0;
            *out_of_bound = 0;
        }else{
            simd_vector_100.copy_from(prev_100, stdx::element_aligned);
        }

        auto prev_011 = iter.prevaddr(out_of_bound,0,1,1);
        if(prev_011 == NULL){
            simd_vector_011 = 0;
        }else if(*out_of_bound == 1){
            simd_vector_011.copy_from(prev_011, stdx::element_aligned);
            simd_vector_011[0] = 0;
            *out_of_bound = 0;
        }else{
            simd_vector_100.copy_from(prev_011, stdx::element_aligned);
        }

        auto prev_101 = iter.prevaddr(out_of_bound,1,0,1);
        if(prev_101 == NULL){
            simd_vector_101 = 0;
        }else if(*out_of_bound == 1){
            simd_vector_101.copy_from(prev_101, stdx::element_aligned);
            simd_vector_101[0] = 0;
            *out_of_bound = 0;
        }else{
            simd_vector_101.copy_from(prev_101, stdx::element_aligned);
        }

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
