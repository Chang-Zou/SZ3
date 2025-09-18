#ifndef _SZ_LINEAR_QUANTIZER_HPP
#define _SZ_LINEAR_QUANTIZER_HPP

#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>
#include <type_traits>
#include <experimental/simd>

#include "SZ3/def.hpp"
#include "SZ3/quantizer/Quantizer.hpp"


namespace stdx {
    using namespace std::experimental;
    using namespace std::experimental::__proposed;
}

namespace SZ3 {


template <class T>
class LinearQuantizer : public concepts::QuantizerInterface<T, int> {
   public:
    LinearQuantizer() : error_bound(1), error_bound_reciprocal(1), radius(32768) {}

    LinearQuantizer(double eb, int r = 32768) : error_bound(eb), error_bound_reciprocal(1.0 / eb), radius(r) {
        assert(eb != 0);
    }

    double get_eb() const { return error_bound; }

    void set_eb(double eb) {
        error_bound = eb;
        error_bound_reciprocal = 1.0 / eb;
    }

    std::pair<int, int> get_out_range() const override { return std::make_pair(0, radius * 2); }

    template <typename TP>
    inline stdx::native_simd<TP> quantize_and_overwrite_simd(const stdx::native_simd<TP> data, const stdx::native_simd<TP> &pred) {
        if constexpr (std::is_same_v<TP, float> || std::is_same_v<TP, double>){
            stdx::native_simd<TP> diff = data - pred; 
            stdx::native_simd_mask<TP> quantizable = (stdx::fabs(diff) < this->radius); 
            auto quant_index = diff + this->radius;
            where(!quantizable,quant_index) *= 0;
            for(std::size_t i=0; i != data.size(); i++){
                if(!quantizable[i]){
                    unpred.push_back(data[i]);
                }
            }
            return quant_index;
        }else{
            stdx::native_simd<TP> diff = data - pred; 
            stdx::native_simd_mask<TP> quantizable = (stdx::abs(diff) < this->radius); 
            auto quant_index = diff + this->radius;
            where(!quantizable,quant_index) *= 0;
            for(std::size_t i=0; i != data.size(); i++){
                if(!quantizable[i]){
                    unpred.push_back(data[i]);
                }
            }
            return  quant_index;
        }

    }
    
    // sequential for simd registers
    inline int quantize_and_overwrite_simd_sequential(T &data, T pred) {
        T diff = data - pred;
        bool quantizable = fabs(diff) < this->radius;
        int quant_index = static_cast<int>(diff + this->radius);
        if(!quantizable){
            unpred.push_back(data);
            quant_index = 0;
        }
        return quant_index;
    }


    // post quant recover back to prequant state
    inline T recover_simd(T pred, int quant_index) {
        if(quant_index){
            return (pred + (quant_index - this->radius));
        }else{
            return recover_unpred(); // return prequant data if false
        }
    }

    // prequant recover back to orig data within errorbound
    template <typename TP>
    inline stdx::native_simd<TP> recover_prequant(stdx::native_simd<TP> pred){
        stdx::native_simd<TP> eb = static_cast<TP>(this->error_bound);
        return (2* eb * pred);
    }

    inline T recover_prequant_sequental(T pred){
        return (2 * this->error_bound * pred);
    }

    inline T recover_unpred() { return unpred[index++]; }

    size_t size_est() { return unpred.size() * sizeof(T); }

    void save(unsigned char *&c) const override {
        // std::string serialized(sizeof(uint8_t) + sizeof(T) + sizeof(int),0);
        c[0] = 0b00000010;
        c += 1;
        *reinterpret_cast<double *>(c) = this->error_bound;
        c += sizeof(double);
        *reinterpret_cast<int *>(c) = this->radius;
        c += sizeof(int);
        *reinterpret_cast<size_t *>(c) = unpred.size();
        c += sizeof(size_t);
        memcpy(c, unpred.data(), unpred.size() * sizeof(T));
        c += unpred.size() * sizeof(T);
    }

    void load(const unsigned char *&c, size_t &remaining_length) override {
        assert(remaining_length > (sizeof(uint8_t) + sizeof(T) + sizeof(int)));
        c += sizeof(uint8_t);
        remaining_length -= sizeof(uint8_t);
        this->error_bound = *reinterpret_cast<const double *>(c);
        this->error_bound_reciprocal = 1.0 / this->error_bound;
        c += sizeof(double);
        this->radius = *reinterpret_cast<const int *>(c);
        c += sizeof(int);
        size_t unpred_size = *reinterpret_cast<const size_t *>(c);
        c += sizeof(size_t);
        this->unpred = std::vector<T>(reinterpret_cast<const T *>(c), reinterpret_cast<const T *>(c) + unpred_size);
        c += unpred_size * sizeof(T);
        // std::cout << "loading: eb = " << this->error_bound << ", unpred_num = "  << unpred.size() << std::endl;
        // reset index
        index = 0;
    }

    void print() override {
        printf("[LinearQuantizer] error_bound = %.8G, radius = %d, unpred = %lu\n", error_bound, radius, unpred.size());
    }

   private:
    std::vector<T> unpred;
    size_t index = 0;  // used in decompression only

    double error_bound;
    double error_bound_reciprocal;
    int radius;  // quantization interval radius
};

}  // namespace SZ3
#endif
