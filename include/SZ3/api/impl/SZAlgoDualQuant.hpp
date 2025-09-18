#ifndef SZ3_SZALGO_DUALQUANT_HPP
#define SZ3_SZALGO_DUALQUANT_HPP

#include <cmath>
#include <memory>

#include "SZ3/compressor/SZSIMDCompressor.hpp" 
#include "SZ3/def.hpp"
#include "SZ3/lossless/Lossless_zstd.hpp"
#include "SZ3/predictor/DualQuantPredictor.hpp" 
#include "SZ3/predictor/LorenzoPredictor.hpp"
#include "SZ3/quantizer/LinearQuantizerDQ.hpp"
#include "SZ3/utils/Config.hpp"
#include "SZ3/utils/Extraction.hpp"
#include "SZ3/utils/Iterator.hpp"
#include "SZ3/utils/QuantOptimizatioin.hpp"
#include "SZ3/utils/Statistic.hpp"

namespace SZ3 {
template <class T, uint N, class Quantizer, class Encoder, class Lossless>
std::shared_ptr<concepts::CompressorInterface<T>> make_compressor_dualquant(const Config &conf, Quantizer quantizer, Encoder encoder, Lossless lossless) {
    if (conf.dualquant) {
        return make_compressor_sz_SIMD<T, N>(conf, DualQuantPredictor<T, N, 1>(conf.absErrorBound), quantizer,
                                                 encoder, lossless);
    }else{
        printf("dualquant methods are disabled.\n");
        exit(0);
    }
}

template <class T, uint N>
size_t SZ_compress_DualQuant(Config &conf, T *data, uchar *cmpData, size_t cmpCap) {
    assert(N == conf.N);
    assert(conf.cmprAlgo == ALGO_DUALQUANT);
    calAbsErrorBound(conf, data);

    auto quantizer = LinearQuantizer<T>(conf.absErrorBound, conf.quantbinCnt / 2);
    auto sz = make_compressor_dualquant<T, N>(conf, quantizer, HuffmanEncoder<int>(), Lossless_zstd());
    return sz->compress(conf, data, cmpData, cmpCap);
}

template <class T, uint N>
void SZ_decompress_DualQuant(const Config &conf, const uchar *cmpData, size_t cmpSize, T *decData) {
    assert(conf.cmprAlgo == ALGO_DUALQUANT);
    auto cmpDataPos = cmpData;
    LinearQuantizer<T> quantizer;
    auto sz = make_compressor_dualquant<T, N>(conf, quantizer, HuffmanEncoder<int>(), Lossless_zstd());
    sz->decompress(conf, cmpDataPos, cmpSize, decData);
}
}  // namespace SZ3
#endif
