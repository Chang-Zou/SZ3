#ifndef _SZ_PREDICTOR_HPP2
#define _SZ_PREDICTOR_HPP2

#include "SZ3/def.hpp"
#include "SZ3/utils/Iterator.hpp"

namespace SZ3::concepts {

/**
 * Data prediction interface
 * Note: You should choose this Predictor interface only when the new prediction method takes scalar value (e.g, *
 * LorenzoPredictor). SZIterateCompressor will automatically iterate through multidimensional data to apply the
 * Predictor on each data point. In all other cases, use the Decomposition interface.
 * @tparam T original data type
 * @tparam N original data dimension
 *
 */
template <class T, uint N>
class PredictorInterface2 {
   public:
    using Range = multi_dimensional_range<T, N>;
    using iterator = typename multi_dimensional_range<T, N>::iterator;

    virtual ~PredictorInterface2() = default;

    virtual void precompress_data(const iterator &) const = 0;

    virtual void postcompress_data(const iterator &) const = 0;

    virtual void predecompress_data(const iterator &) const = 0;

    virtual void postdecompress_data(const iterator &) const = 0;

    /**
     * compute auxiliary info (e.g., coefficients) for the given data block
     * @param iterator of the block
     * @return whether the predictor is suitable for the block (e.g., data with 100x1 shape is not suitable for 2D
     * regression)
     */

    virtual bool precompress_block(const std::shared_ptr<Range> &block) = 0;

    /**
     * store the auxiliary info (e.g., coefficients) to this class's internal storage
     */
    virtual void precompress_block_commit() = 0;

    virtual bool predecompress_block(const std::shared_ptr<Range> &) = 0;

    /**
     * serialize the predictor and store it to a buffer
     * @param c One large buffer is pre-allocated, and the start location of the serialized predictor in the buffer is
     * indicated by c. After saving the predictor to the buffer, this function should change c to indicate the next
     * empty location in the buffer
     */
    virtual void save(uchar *&c) const = 0;

    /**
     * deserialize the predictor from a buffer
     * @param c start location of the predictor in the buffer
     * @param remaining_length the remaining length of the buffer
     */
    virtual void load(const uchar *&c, size_t &remaining_length) = 0;

    /**
     * predict the value for a single data point
     * @param iter the iterator of the single data point
     * @return the predicted value
     */
    virtual T predict(const iterator &iter) const noexcept = 0;

    virtual T simd_prequant(iterator &iter) noexcept {
        // Provide a default implementation if not overridden
        throw std::runtime_error("simdpredict is not implemented for this predictor.");
    }

    /**
     * estimate the prediction error ( |prediction value - read value|)  for a single data point
     * @param iter the iterator of the single data point
     * @return the estimated prediction error
     */
    virtual T estimate_error(const iterator &iter) const noexcept = 0;

    virtual void print() const = 0;

    //        virtual void clear() = 0;
};

/**
 * Concept for the predictor class.
 *
 * Matches classes like the following:
 *
 * class my_predictor {
 *  using iterator = std::multi_dimensional_range<T,N>::iterator
 *
 *  /// returns the prediction for the single element pointed to by the iterator
 *  T predict(const iterator);
 *
 *  /// pre-processing hook run at the beginning of compression
 *  void precompress_data(const iterator);
 *
 *  /// post-processing hook run at the end of compression
 *  void postcompress_data(const iterator);
 *
 *  /// pre-processing hook run before compressing each block
 *  /// post processing can be done either during postcompress_data or on the next call to precompress_block
 *  void precompress_block(const iterator);
 *
 *  /// pre-processing hook run at the beginning of decompression
 *  void predecompress_data(const iterator);
 *
 *  /// pre-processing hook run at the end of decompression
 *  void postdecompress_data(const iterator);
 *
 *  /// pre-processing hook run before decompressing each block
 *  /// post processing can be done either during postcompress_data or on the next call to precompress_block
 *  void predecompress_block(const iterator);
 *
 *  /// returns a string which represents the configuration of the preditor in a serialized form
 *  std::string save() const;
 *
 *  /// returns a predictor from a serialized form
 *  static my_predictor load(const unsigned char*&, size_t& len);
 *
 * };
 */

//        template<typename T, typename = void>
//        struct is_predictor : false_type {
//        };
//        template<typename T>
//        struct is_predictor<T, void_t<
//                typename T::iterator,
//                decltype(std::declval<T>().predict(std::declval<typename T::iterator const>())),
//                decltype(std::declval<T>().precompress_data(std::declval<typename T::iterator const>())),
//                decltype(std::declval<T>().postcompress_data(std::declval<typename T::iterator const>())),
//                decltype(std::declval<T>().predecompress_data(std::declval<typename T::iterator const>())),
//                decltype(std::declval<T>().postdecompress_data(std::declval<typename T::iterator const>())),
//                // TODO: make the interface for precompress_block
//                // decltype(std::declval<T>().precompress_block(std::declval<typename T::std::shared_ptr<Range>
//                const>())),
//                // decltype(std::declval<T>().predecompress_block(std::declval<typename T::std::shared_ptr<Range>
//                const>())),
//                // decltype(std::declval<T>().save()),
//                // decltype(T::load(std::declval<const unsigned char*&>(), std::declval<size_t&>()))
//                decltype(std::declval<T>().save(
//                        std::declval<unsigned char *&>())
//                ),
//                decltype(std::declval<T>().load(
//                        std::declval<const unsigned char *&>(), std::declval<size_t &>())
//                )
//        >> : true_type {
//            //we must remove_reference otherwise we get the const-ness of the reference not the underlying type
//            // static_assert(
//            //   std::is_const<typename std::remove_reference<
//            //       decltype(*std::declval<typename T::iterator const>())
//            //     >::type
//            //   >::value, "const iterators must not be writable");
//            static_assert(
//                    std::is_same<
//                            typename std::iterator_traits<typename T::iterator>::value_type,
//                            decltype(std::declval<T>().predict(std::declval<typename T::iterator const>()))
//                    >::value, "predict must return iterator's value type"
//            );
//            // static_assert(
//            //     std::is_same<
//            //       decltype(std::declval<T>().save()),
//            //       std::string
//            //     >::value, "save must return a string");
//        };

}  // namespace SZ3::concepts

#endif
