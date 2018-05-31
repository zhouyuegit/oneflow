//
// Created by root on 5/27/18.
//

#ifndef ONEFLOW_META_UTIL_HPP
#define ONEFLOW_META_UTIL_HPP

namespace oneflow{
    template <typename T, T... Idx>
    struct integer_sequence{
        static_assert(std::is_integral<T>::value,
                      "integer_sequence<T, I...> requires T to be an integral type.");
        using value_type = T;
        static constexpr std::size_t size() noexcept { return sizeof...(Idx); }
    };

    template<size_t... Idx>
    using index_sequence = integer_sequence<size_t, Idx...>;

    template<size_t N, size_t... Idx>
    struct make_index_sequence_impl : make_index_sequence_impl<N-1, N-1, Idx...>{};

    template <size_t... Idx>
    struct make_index_sequence_impl<0, Idx...>{
        using type = index_sequence<Idx...>;
    };

    template<size_t N>
    using make_index_sequence = typename make_index_sequence_impl<N>::type;

    template <typename... Args, typename Func, std::size_t... Idx>
    void for_each(const std::tuple<Args...>& t, Func&& f, index_sequence<Idx...>) {
        (void)std::initializer_list<int> { (f(std::get<Idx>(t)), void(), 0)...};
    }

    template <typename... Args, typename Func, std::size_t... Idx>
    void for_each_i(const std::tuple<Args...>& t, Func&& f, index_sequence<Idx...>) {
        (void)std::initializer_list<int> { (f(std::get<Idx>(t), std::integral_constant<size_t, Idx>{}), void(), 0)...};
    }

    template <class T, class Tuple>
    struct index_of;

    template <class T, class... Types>
    struct index_of<T, std::tuple<T, Types...>> {
        static const std::size_t value = 0;
    };

    template <class T, class U, class... Types>
    struct index_of<T, std::tuple<U, Types...>> {
        static const std::size_t value = 1 + index_of<T, std::tuple<Types...>>::value;
    };

    template<typename>
    struct array_size;

    template<typename T, size_t N>
    struct array_size<std::array<T,N> > {
        static size_t const size = N;
    };

    template<typename T>
    struct function_traits;

    template<typename Ret, typename... Args>
    struct function_traits<Ret(Args...)>
    {
    public:
        enum { arity = sizeof...(Args) };
        typedef Ret function_type(Args...);
        typedef Ret return_type;
        using stl_function_type = std::function<function_type>;
        typedef Ret(*pointer)(Args...);

        typedef std::tuple<Args...> tuple_type;
    };

    template<typename Ret, typename... Args>
    struct function_traits<Ret(*)(Args...)> : function_traits<Ret(Args...)>{};

    template <typename Ret, typename... Args>
    struct function_traits<std::function<Ret(Args...)>> : function_traits<Ret(Args...)>{};

    template <typename ReturnType, typename ClassType, typename... Args>
    struct function_traits<ReturnType(ClassType::*)(Args...)> : function_traits<ReturnType(Args...)>{};

    template <typename ReturnType, typename ClassType, typename... Args>
    struct function_traits<ReturnType(ClassType::*)(Args...) const> : function_traits<ReturnType(Args...)>{};

    template<typename Callable>
    struct function_traits : function_traits<decltype(&Callable::operator())>{};
}

#endif //ONEFLOW_META_UTIL_HPP
