#ifndef SZ_COMPRESSOR_DUAL_QUANT_HPP
#define SZ_COMPRESSOR_DUAL_QUANT_HPP

#include <cstring>
#include <experimental/simd>// simd

#include "SZ3/compressor/Compressor.hpp"
#include "SZ3/def.hpp"
#include "SZ3/encoder/Encoder.hpp"
#include "SZ3/lossless/Lossless.hpp"
#include "SZ3/predictor/LorenzoPredictor.hpp"
#include "SZ3/predictor/Predictor.hpp"
#include "SZ3/quantizer/Quantizer.hpp"
#include "SZ3/utils/Config.hpp"
#include "SZ3/utils/FileUtil.hpp"
#include "SZ3/utils/Iterator.hpp"
#include "SZ3/utils/MemoryUtil.hpp"
#include "SZ3/utils/Timer.hpp"

namespace stdx {
    using namespace std::experimental;
    using namespace std::experimental::__proposed;
}

//namespace stdx = vir::stdx; // works require c++20


namespace SZ3 {
/**
 * SZSIMDCompressor glues together predictor, quantizer, encoder, and lossless modules to form the compressor.
 * It only takes Predictor, not Decomposition.
 * It will automatically iterate through multidimensional data to apply the Predictor.
 *
 * @tparam T original data type
 * @tparam N original data dimension
 * @tparam Predictor    predictor module
 * @tparam Quantizer   quantizer module
 * @tparam Encoder    encoder module
 * @tparam Lossless  lossless module
 */
template <class T, uint N, class Predictor, class Quantizer, class Encoder, class Lossless>
class SZSIMDCompressor : public concepts::CompressorInterface<T> {
   public:
    SZSIMDCompressor(const Config &conf, Predictor predictor, Quantizer quantizer, Encoder encoder,
                         Lossless lossless)
        : fallback_predictor(LorenzoPredictor<T, N, 1>(conf.absErrorBound)),
          predictor(predictor),
          quantizer(quantizer),
          block_size(conf.blockSize),
          num_elements(conf.num),
          encoder(encoder),
          lossless(lossless) {
        std::copy_n(conf.dims.begin(), N, global_dimensions.begin());
        static_assert(std::is_base_of<concepts::PredictorInterface<T, N>, Predictor>::value,
                      "must implement the predictor interface");
        static_assert(std::is_base_of<concepts::QuantizerInterface<T, int>, Quantizer>::value,
                      "must implement the quantizer interface");
        static_assert(std::is_base_of<concepts::EncoderInterface<int>, Encoder>::value,
                      "must implement the encoder interface");
        static_assert(std::is_base_of<concepts::LosslessInterface, Lossless>::value,
                      "must implement the lossless interface");
    }

    size_t compress(const Config &conf, T *data, uchar *cmpData, size_t cmpCap) override {
        std::vector<int> quant_inds(num_elements);
        auto block_range = std::make_shared<multi_dimensional_range<T, N>>(data, std::begin(global_dimensions),
                                                                           std::end(global_dimensions), block_size, 0);

        auto element_range = std::make_shared<multi_dimensional_range<T, N>>(data, std::begin(global_dimensions),
                                                                             std::end(global_dimensions), 1, 0);

        predictor.precompress_data(block_range->begin());
        quantizer.precompress_data();
        size_t quant_count = 0;
        size_t batch_size = stdx::native_simd<T>::size();
        stdx::native_simd<T> orig_element;

        // Pre-quant maybe we don't even need to do this block by block?
        for (auto block = block_range->begin(); block != block_range->end(); ++block) {
            element_range->update_block_range(block, block_size);

            // This interface doesn't know the existance of dualquant.
            concepts::PredictorInterface<T, N> *predictor_withfallback = &predictor;
            if (!predictor.precompress_block(element_range)) {
                predictor_withfallback = &fallback_predictor;
            }
            predictor_withfallback->precompress_block_commit();

            size_t element_size = element_range->end().get_offset() - element_range->begin().get_offset();
            auto element = element_range->begin();

            if(N == 1){
                if(element_size % batch_size == 0){ // fits in SIMD register completely
                    for(; element != element_range->end(); element += batch_size) {
                        orig_element.copy_from(&(*element),stdx::element_aligned);
                        auto temp_storage = quantizer.quantize_and_overwrite_simd(orig_element, predictor.simd_predict(element));
                        temp_storage.copy_to(&quant_inds[quant_count],stdx::element_aligned);
                        quant_count = quant_count + batch_size;
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
                        orig_element.copy_from(&(*element),stdx::element_aligned);
                        auto temp_storage = quantizer.quantize_and_overwrite_simd(orig_element, predictor.simd_predict(element));
                        temp_storage.copy_to(&quant_inds[quant_count],stdx::element_aligned);
                        quant_count = quant_count + batch_size;
                        count +=batch_size;
                    }
                    for(; element != element_range->end(); ++element) {
                        quant_inds[quant_count++] =
                            quantizer.quantize_and_overwrite_simd_sequential(*element, predictor_withfallback->predict(element));
                    }
                }
            }else if (N == 2){ //N == 2
                auto row = element.get_dimensions()[0];
                auto col = element.get_dimensions()[1];
                if((element_size % batch_size == 0) && (col % batch_size == 0)){ // fits in SIMD register completely
                    for(; element != element_range->end(); element += batch_size) {
                        orig_element.copy_from(&(*element),stdx::element_aligned);
                        auto temp_storage = quantizer.quantize_and_overwrite_simd(orig_element, predictor.simd_predict(element));
                        temp_storage.copy_to(&quant_inds[quant_count],stdx::element_aligned);
                        quant_count = quant_count + batch_size;
                    }
                }else if (col > batch_size && col % batch_size !=0) {
                    while(element != element_range->end()){
                        size_t count = 0;
                        for(; count + batch_size < col; element+=batch_size){
                            orig_element.copy_from(&(*element),stdx::element_aligned);
                            auto temp_storage = quantizer.quantize_and_overwrite_simd(orig_element, predictor.simd_predict(element));
                            temp_storage.copy_to(&quant_inds[quant_count],stdx::element_aligned);
                            quant_count = quant_count + batch_size;
                            count +=batch_size;
                        }
                        for(; count < col; ++element){
                            quant_inds[quant_count++] =
                                quantizer.quantize_and_overwrite_simd_sequential(*element, predictor_withfallback->predict(element));
                            count++;
                        }
                    }
                }else{
                    for(; element != element_range->end(); ++element) {
                        quant_inds[quant_count++] =
                            quantizer.quantize_and_overwrite_simd_sequential(*element, predictor_withfallback->predict(element));
                    }
                }
            }else{
                auto depth = element.get_dimensions()[0];
                auto row = element.get_dimensions()[1];
                auto col = element.get_dimensions()[2];
                if((element_size % batch_size == 0) && (col % batch_size == 0)){ // fits in SIMD register completely
                    for(; element != element_range->end(); element += batch_size) {
                        orig_element.copy_from(&(*element),stdx::element_aligned);
                        auto temp_storage = quantizer.quantize_and_overwrite_simd(orig_element, predictor.simd_predict(element));
                        temp_storage.copy_to(&quant_inds[quant_count],stdx::element_aligned);
                        quant_count = quant_count + batch_size;
                    }
                }else if (col > batch_size && col % batch_size !=0) {
                    while(element != element_range->end()){
                        size_t count = 0;
                        for(; count + batch_size < col; element+=batch_size){
                            orig_element.copy_from(&(*element),stdx::element_aligned);
                            auto temp_storage = quantizer.quantize_and_overwrite_simd(orig_element, predictor.simd_predict(element));
                            temp_storage.copy_to(&quant_inds[quant_count],stdx::element_aligned);
                            quant_count = quant_count + batch_size;
                            count +=batch_size;
                        }
                        for(; count < col; ++element){
                            quant_inds[quant_count++] =
                                quantizer.quantize_and_overwrite_simd_sequential(*element, predictor_withfallback->predict(element));
                            count++;
                        }
                    }
                }else{
                    for(; element != element_range->end(); ++element) {
                        quant_inds[quant_count++] =
                            quantizer.quantize_and_overwrite_simd_sequential(*element, predictor_withfallback->predict(element));
                    }
                }
            }
        }


        predictor.postcompress_data(block_range->begin());
        quantizer.postcompress_data();

        if (quantizer.get_out_range().first != 0) {
            throw std::runtime_error("The output range of the quantizer must start from 0 for this compressor");
        }
        encoder.preprocess_encode(quant_inds, quantizer.get_out_range().second);

        size_t bufferSize = 1.2 * (quantizer.size_est() + encoder.size_est() + sizeof(T) * quant_inds.size());
        auto buffer = static_cast<uchar *>(malloc(bufferSize));
        uchar *buffer_pos = buffer;

        write(conf.num, buffer_pos);
        write(global_dimensions.data(), N, buffer_pos);
        write(block_size, buffer_pos);

        predictor.save(buffer_pos);
        quantizer.save(buffer_pos);

        encoder.save(buffer_pos);
        encoder.encode(quant_inds, buffer_pos);
        encoder.postprocess_encode();

        // assert(buffer_pos - buffer < bufferSize);

        auto cmpSize = lossless.compress(buffer, buffer_pos - buffer, cmpData, cmpCap);
        free(buffer);
        return cmpSize;
    }

    T *decompress(const Config &conf, uchar const *cmpData, size_t cmpSize, T *decData) override {
        //            Timer timer(true);
        uchar *buffer = nullptr;
        size_t bufferSize = 0;
        lossless.decompress(cmpData, cmpSize, buffer, bufferSize);
        size_t remaining_length = bufferSize;
        uchar const *buffer_pos = buffer;
        //            timer.stop("Lossless");
        size_t num = 0;
        read(num, buffer_pos, remaining_length);

        read(global_dimensions.data(), N, buffer_pos, remaining_length);
        num_elements = 1;
        for (const auto &d : global_dimensions) {
            num_elements *= d;
        }
        read(block_size, buffer_pos, remaining_length);
        predictor.load(buffer_pos, remaining_length);
        quantizer.load(buffer_pos, remaining_length);

        encoder.load(buffer_pos, remaining_length);

        //            timer.start();
        auto quant_inds = encoder.decode(buffer_pos, num);
        encoder.postprocess_decode();
        //            timer.stop("Decoder");

        free(buffer);
        //            lossless.postdecompress_data(buffer);

        //            timer.start();
        int const *quant_inds_pos = static_cast<int const *>(quant_inds.data());
        // std::array<size_t, N> intra_block_dims;
        auto block_range = std::make_shared<multi_dimensional_range<T, N>>(decData, std::begin(global_dimensions),
                                                                           std::end(global_dimensions), block_size, 0);

        auto element_range = std::make_shared<multi_dimensional_range<T, N>>(decData, std::begin(global_dimensions),
                                                                             std::end(global_dimensions), 1, 0);

        predictor.predecompress_data(block_range->begin());
        quantizer.predecompress_data();

        for (auto block = block_range->begin(); block != block_range->end(); ++block) {
            element_range->update_block_range(block, block_size);

            concepts::PredictorInterface<T, N> *predictor_withfallback = &predictor;
            if (!predictor.predecompress_block(element_range)) {
                predictor_withfallback = &fallback_predictor;
            }
            for (auto element = element_range->begin(); element != element_range->end(); ++element) {
                *element = quantizer.recover_simd(predictor_withfallback->predict(element), *(quant_inds_pos++));
            }
        }

        size_t quant_count = 0;
        size_t batch_size = stdx::native_simd<T>::size();
        stdx::native_simd<T> orig_element;

        for (auto block = block_range->begin(); block != block_range->end(); ++block) {
            element_range->update_block_range(block, block_size);
            auto element = element_range->begin();
            size_t element_size = element_range->end().get_offset() - element_range->begin().get_offset();

            if(N == 1){
                if(element_size % batch_size == 0){
                    for (; element != element_range->end(); element += batch_size) {
                        orig_element.copy_from(&(*element),stdx::element_aligned);
                        auto temp_storage = quantizer.recover_prequant(orig_element);
                        temp_storage.copy_to(&(*element += quant_count), stdx::element_aligned);
                        quant_count = quant_count+ batch_size;
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
                    for (; count < element_end; element += batch_size) {
                        orig_element.copy_from(&(*element),stdx::element_aligned);
                        auto temp_storage = quantizer.recover_prequant(orig_element);
                        temp_storage.copy_to(&(*element += quant_count), stdx::element_aligned);
                        quant_count = quant_count+ batch_size;
                        count +=batch_size;
                    }
                    for (; element != element_range->end(); ++element) {
                        *element = quantizer.recover_prequant_sequental(*element);
                    }
                }
            }else if (N == 2){ // N == 2
                auto row = element.get_dimensions()[0];
                auto col = element.get_dimensions()[1];
                if((element_size % batch_size == 0) && (col % batch_size == 0)){
                    for (; element != element_range->end(); element += batch_size) {
                        orig_element.copy_from(&(*element),stdx::element_aligned);
                        auto temp_storage = quantizer.recover_prequant(orig_element);
                        temp_storage.copy_to(&(*element += quant_count), stdx::element_aligned);
                        quant_count = quant_count+ batch_size;
                    }
                }else if(col > batch_size && col % batch_size !=0){
                    while(element != element_range->end()){
                        size_t count = 0;
                        for(; count + batch_size < col; element+=batch_size){
                            orig_element.copy_from(&(*element),stdx::element_aligned);
                            auto temp_storage = quantizer.recover_prequant(orig_element);
                            temp_storage.copy_to(&(*element += quant_count), stdx::element_aligned);
                            quant_count = quant_count+ batch_size;
                            count +=batch_size;
                        }
                        for(; count < col; ++element){
                            *element = quantizer.recover_prequant_sequental(*element);
                            count++;
                        }
                    }
                }else{
                    for (; element != element_range->end(); ++element) {*element = quantizer.recover_prequant_sequental(*element);}
                }
            }else{
                auto depth = element.get_dimensions()[0];
                auto row = element.get_dimensions()[1];
                auto col = element.get_dimensions()[2];
                if((element_size % batch_size == 0) && (col % batch_size == 0)){
                    for (; element != element_range->end(); element += batch_size) {
                        orig_element.copy_from(&(*element),stdx::element_aligned);
                        auto temp_storage = quantizer.recover_prequant(orig_element);
                        temp_storage.copy_to(&(*element += quant_count), stdx::element_aligned);
                        quant_count = quant_count+ batch_size;
                    }
                }else if(col > batch_size && col % batch_size !=0){
                    while(element != element_range->end()){
                        size_t count = 0;
                        for(; count + batch_size < col; element+=batch_size){
                            orig_element.copy_from(&(*element),stdx::element_aligned);
                            auto temp_storage = quantizer.recover_prequant(orig_element);
                            temp_storage.copy_to(&(*element += quant_count), stdx::element_aligned);
                            quant_count = quant_count+ batch_size;
                            count +=batch_size;
                        }
                        for(; count < col; ++element){
                            *element = quantizer.recover_prequant_sequental(*element);
                            count++;
                        }
                    }
                }else{
                    for (; element != element_range->end(); ++element) {*element = quantizer.recover_prequant_sequental(*element);}
                }
            }
        }

        predictor.postdecompress_data(block_range->begin());
        quantizer.postdecompress_data();
        return decData;
        //            timer.stop("Prediction & Recover");
    }

   private:
    Predictor predictor;
    LorenzoPredictor<T, N, 1> fallback_predictor;
    Quantizer quantizer;
    uint block_size;
    size_t num_elements;
    std::array<size_t, N> global_dimensions;
    Encoder encoder;
    Lossless lossless;
};

template <class T, uint N, class Predictor, class Quantizer, class Encoder, class Lossless>
std::shared_ptr<SZSIMDCompressor<T, N, Predictor, Quantizer, Encoder, Lossless>> make_compressor_sz_SIMD(
    const Config &conf, Predictor predictor, Quantizer quantizer, Encoder encoder, Lossless lossless) {
    return std::make_shared<SZSIMDCompressor<T, N, Predictor, Quantizer, Encoder, Lossless>>(
        conf, predictor, quantizer, encoder, lossless);
}

}  // namespace SZ3
#endif
