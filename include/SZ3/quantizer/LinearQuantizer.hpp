#ifndef _SZ_LINEAR_QUANTIZER_HPP
#define _SZ_LINEAR_QUANTIZER_HPP

#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>
#include <type_traits>
#include <experimental/simd>
//#include </home/changfz/.local/include/vir/simd.h> // direct path need change
//#include </home/changfz/.local/include/vir/simd_cvt.h> // direct path need change
//#include </home/changfz/.local/include/vir/simd_resize.h> // direct path need change

#include "SZ3/def.hpp"
#include "SZ3/quantizer/Quantizer.hpp"

//namespace stdx = vir::stdx; // works require c++20


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

    int quantize_and_overwrite(T &data, T pred) override {
        // int quantize_and_overwrite(T &data, const stdx::native_simd<float> &simd_vector) override {
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
            if (fabs(decompressed_data - data) > this->error_bound) {
                unpred.push_back(data);
                return 0;
            } else {
                data = decompressed_data;
                return quant_index_shifted;
            }
        } else {
            unpred.push_back(data);
            return 0;
        }
    }
    void printsimd(auto const &a) {
        for (std::size_t i{}; i != std::size(a); ++i) std::cout << a[i] << ' ';
        std::cout << '\n';
    }
  
    /*
    fabs wrapper function
    */
    template <typename Q>
    Q my_fabs(Q value){
        using valueType = typename Q::value_type;

        if constexpr (std::is_same_v<valueType, float> || std::is_same_v<valueType, double>){
            return stdx::fabs(value);
        }else{
            return 0;
        }
    }
    

    template <typename TP>
    stdx::native_simd<TP> quantize_and_overwrite2(const stdx::native_simd<TP> data, const stdx::native_simd<TP> &pred) { 
        stdx::native_simd<TP> diff = data - pred; // find the difference
        //stdx::native_simd<TP> eb = static_cast<TP>(this->error_bound*2); 
        //stdx::native_simd<TP> radius = static_cast<TP>(this->radius *2); // grab radius -> simd register
        //stdx::native_simd<TP> radiusX2 = radius*2; // 2x radius

        //using floatv = stdx::native_simd<TP>;
        using maskv = stdx::native_simd_mask<TP>;
        //using maskintv = stdx::rebind_simd_t<int, maskv>;
        //using intv = stdx::rebind_simd_t<int, floatv>;

        maskv quantizable = (my_fabs(diff) < this->radius); // are you quantizable? in float
        //maskintv quant_int = stdx::static_simd_cast<maskintv>(quantizable);
        auto quant_index = diff + radius; // _code in float
        //auto _code = stdx::static_simd_cast<intv>(_codefloat); // int code representation of the float values

        //stdx::native_simd<TP> outlier = data;
        //where(quantizable,outlier) *= 0; // if quantizable then it is not outlier
        where(!quantizable,quant_index) *= 0;
        for(std::size_t i=0; i != data.size(); i++){
            if(!quantizable[i]){
                unpred.push_back(data[i]);
            }
        }

        return  quant_index;
    }

    /**
     * For metaLorenzo only, will be removed together with metalorenzo
     * @param ori
     * @param pred
     * @param dest
     * @return
     */
    int quantize_and_overwrite(T ori, T pred, T &dest) {
        T diff = ori - pred;
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
            if (fabs(decompressed_data - ori) > this->error_bound) {
                unpred.push_back(ori);
                dest = ori;
                return 0;
            } else {
                dest = decompressed_data;
                return quant_index_shifted;
            }
        } else {
            unpred.push_back(ori);
            dest = ori;
            return 0;
        }
    }

    // recover the data using the quantization index
    T recover(T pred, int quant_index) override {
        if (quant_index) {
            return recover_pred(pred, quant_index);
        } else {
            return recover_unpred();
        }
    }

    // error bound double in results
    // T simdrecover(T pred, int quant_index) {
    //     if (quant_index) {
    //         return recover_pred(pred, quant_index);
    //     } else {
    //         return recover_dualunpred();
    //     }
    // }
    
    // post quant recover back to prequant state
    T recoverPostQ(T pred, int quant_index) {
        if(quant_index){
            return (pred + (quant_index - this->radius));
        }else{
            return recover_unpred(); // return prequant data if false
        }
    }

    // prequant recover back to orig data within errorbound
    T PreQrecover(T pred){
        return (2 * this->error_bound * pred);
    }

    T recover_pred(T pred, int quant_index) { return pred + 2 * (quant_index - this->radius) * this->error_bound; }

    T recover_unpred() { return unpred[index++]; }

    //T recover_dualunpred() { return  2 * this->error_bound * unpred[index++]; } errors

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

    //        void clear() {
    //            unpred.clear();
    //            index = 0;
    //        }

   private:
    std::vector<T> unpred;
    size_t index = 0;  // used in decompression only

    double error_bound;
    double error_bound_reciprocal;
    int radius;  // quantization interval radius
};

}  // namespace SZ3
#endif
