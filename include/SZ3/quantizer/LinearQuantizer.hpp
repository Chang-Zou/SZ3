#ifndef SZ3_LINEAR_QUANTIZER_HPP
#define SZ3_LINEAR_QUANTIZER_HPP

#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

#include <type_traits>
#include <experimental/simd>

#include "SZ3/def.hpp"
#include "SZ3/quantizer/Quantizer.hpp"
#include "SZ3/utils/MemoryUtil.hpp"


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

    // quantize the data with a prediction value, and returns the quantization index and the decompressed data
    // int quantize(T data, T pred, T& dec_data);
    ALWAYS_INLINE int quantize_and_overwrite(T &data, T pred) override {
        T diff = data - pred;
        auto quant_index = static_cast<int64_t>(fabs(diff) * this->error_bound_reciprocal) + 1;
        if (quant_index < this->radius * 2) {
            quant_index >>= 1;
            int half_index = quant_index;
            quant_index <<= 1;
            int quant_index_shifted;
            if (diff < 0) {
                quant_index = -quant_index;
                quant_index_shifted = this->radius - half_index;
            } else {
                quant_index_shifted = this->radius + half_index;
            }
            T decompressed_data = pred + quant_index * this->error_bound;
            // if data is NaN, the error is NaN, and NaN <= error_bound is false
            if (fabs(decompressed_data - data) <= this->error_bound) {
                data = decompressed_data;
                return quant_index_shifted;
            } else {
                unpred.push_back(data);
                return 0;
            }
        } else {
            unpred.push_back(data);
            return 0;
        }
    }

    // Dual Quant methods of quantization process (SIMD)
    template <typename TP>
    ALWAYS_INLINE stdx::native_simd<TP> quantize_and_overwrite_simd(const stdx::native_simd<TP> data, const stdx::native_simd<TP> &pred) {
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
    
    // Dual Quant methods of quantization process (sequential) 
    ALWAYS_INLINE int quantize_and_overwrite_simd_sequential(T &data, T pred) {
        T diff = data - pred;
        bool quantizable = fabs(diff) < this->radius;
        int quant_index = static_cast<int>(diff + this->radius);
        if(!quantizable){
            unpred.push_back(data);
            quant_index = 0;
        }
        return quant_index;
    }

    // recover the data using the quantization index
    ALWAYS_INLINE T recover(T pred, int quant_index) override {
        if (quant_index) {
            return recover_pred(pred, quant_index);
        } else {
            return recover_unpred();
        }
    }

    // post quant recover back to prequant state
    ALWAYS_INLINE T recover_simd(T pred, int quant_index) {
        if(quant_index){
            return (pred + (quant_index - this->radius));
        }else{
            return recover_unpred(); // return prequant data if false
        }
    }

    // prequant recover back to orig data within errorbound (SIMD)
    template <typename TP>
    ALWAYS_INLINE stdx::native_simd<TP> recover_prequant(stdx::native_simd<TP> pred){
        stdx::native_simd<TP> eb = static_cast<TP>(this->error_bound);
        return (2* eb * pred);
    }

    // prequant recover back to orig data within errorbound (sequential)
    ALWAYS_INLINE T recover_prequant_sequential(T pred){
        return (2 * this->error_bound * pred);
    }

    ALWAYS_INLINE T recover_pred(T pred, int quant_index) {
        return pred + 2 * (quant_index - this->radius) * this->error_bound;
    }

    ALWAYS_INLINE T recover_unpred() { return unpred[index++]; }

    ALWAYS_INLINE int force_save_unpred(T ori) override {
        unpred.push_back(ori);
        return 0;
    }

    size_t size_est() { return unpred.size() * sizeof(T); }

    void save(unsigned char *&c) const override {
        write(uid, c);
        write(this->error_bound, c);
        write(this->radius, c);
        size_t unpred_size = unpred.size();
        write(unpred_size, c);
        if (unpred_size > 0) {
            write(unpred.data(), unpred.size(), c);
        }
    }

    void load(const unsigned char *&c, size_t &remaining_length) override {
        uchar uid_read;
        read(uid_read, c, remaining_length);
        if (uid_read != uid) {
            throw std::invalid_argument("LinearQuantizer uid mismatch");
        }
        read(this->error_bound, c, remaining_length);
        this->error_bound_reciprocal = 1.0 / this->error_bound;
        read(this->radius, c, remaining_length);
        size_t unpred_size = 0;
        read(unpred_size, c, remaining_length);
        if (unpred_size > 0) {
            unpred.resize(unpred_size);
            read(unpred.data(), unpred_size, c, remaining_length);
        }
        index = 0;
    }

    void print() override {
        printf("[LinearQuantizer] error_bound = %.8G, radius = %d, unpred = %zu\n", error_bound, radius, unpred.size());
    }

   private:
    std::vector<T> unpred;
    size_t index = 0;  // used in decompression only
    uchar uid = 0b10;

    double error_bound;
    double error_bound_reciprocal;
    int radius;  // quantization interval radius
};

}  // namespace SZ3
#endif