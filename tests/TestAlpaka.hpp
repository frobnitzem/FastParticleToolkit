#pragma once

#include "Check.hpp"
//#include <alpaka/test/test.hpp>
namespace alpaka { namespace test {
using TestDims = std::tuple< alpaka::DimInt<1u> >;
using TestIdxs = std::tuple< std::uint32_t >;
}; };
#include "TestAccs.hpp"
#include "KernelExecutionFixture.hpp"
