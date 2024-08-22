//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//


#pragma once

#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <optix.h>

#include <glad/glad.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

//------------------------------------------------------------------------------
//
// GL error-checking
//
//------------------------------------------------------------------------------

#define DO_GL_CHECK
#ifdef DO_GL_CHECK

#define GL_CHECK( call )                                                       \
    do                                                                         \
    {                                                                          \
        call;                                                                  \
        ::sutil::glCheck( #call, __FILE__, __LINE__ );                         \
    } while( false )


#define GL_CHECK_ERRORS() ::sutil::glCheckErrors( __FILE__, __LINE__ )

#else
#define GL_CHECK( call )                                                       \
    do                                                                         \
    {                                                                          \
        call;                                                                  \
    } while( 0 )
#define GL_CHECK_ERRORS()                                                      \
    do                                                                         \
    {                                                                          \
        ;                                                                      \
    } while( 0 )
#endif


//------------------------------------------------------------------------------
//
// OptiX error-checking
//
//------------------------------------------------------------------------------

#define OPTIX_CHECK( call )                                                    \
    ::sutil::optixCheck( call, #call, __FILE__, __LINE__ )

// This version of the log-check macro doesn't require the user do setup
// a log buffer and size variable in the surrounding context; rather the
// macro defines a log buffer and log size variable (LOG and LOG_SIZE)
// respectively that should be passed to the message being checked.
// E.g.:
//  OPTIX_CHECK_LOG2( optixProgramGroupCreate( ..., LOG, &LOG_SIZE, ... );
//
#define OPTIX_CHECK_LOG( call )                                                \
    do                                                                         \
    {                                                                          \
        char   LOG[2048];                                                      \
        size_t LOG_SIZE = sizeof( LOG );                                       \
        ::sutil::optixCheckLog( call, LOG, sizeof( LOG ), LOG_SIZE, #call,     \
                                __FILE__, __LINE__ );                          \
    } while( false )

#define OPTIX_CHECK_NOTHROW( call )                                            \
    ::sutil::optixCheckNoThrow( call, #call, __FILE__, __LINE__ )

//------------------------------------------------------------------------------
//
// CUDA error-checking
//
//------------------------------------------------------------------------------

#define CUDA_CHECK( call ) ::sutil::cudaCheck( call, #call, __FILE__, __LINE__ )

#define CUDA_SYNC_CHECK() ::sutil::cudaSyncCheck( __FILE__, __LINE__ )

// A non-throwing variant for use in destructors.
// An iostream must be provided for output (e.g. std::cerr).
#define CUDA_CHECK_NOTHROW( call )                                             \
    ::sutil::cudaCheckNoThrow( call, #call, __FILE__, __LINE__ )

//------------------------------------------------------------------------------
//
// Assertions
//
//------------------------------------------------------------------------------

#define SUTIL_ASSERT( cond )                                                   \
    ::sutil::assertCond( static_cast<bool>( cond ), #cond, __FILE__, __LINE__ )

#define SUTIL_ASSERT_MSG( cond, msg )                                          \
    ::sutil::assertCondMsg( static_cast<bool>( cond ), #cond, msg, __FILE__, __LINE__ )

#define SUTIL_ASSERT_FAIL_MSG( msg )                                           \
    ::sutil::assertFailMsg( msg, __FILE__, __LINE__ )

namespace sutil {

class Exception : public std::runtime_error
{
  public:
    Exception( const char* msg )
        : std::runtime_error( msg )
    {
    }

    Exception( OptixResult res, const char* msg )
        : std::runtime_error( createMessage( res, msg ).c_str() )
    {
    }

  private:
    std::string createMessage( OptixResult res, const char* msg )
    {
        std::ostringstream out;
        out << optixGetErrorName( res ) << ": " << msg;
        return out.str();
    }
};

inline void optixCheck( OptixResult res, const char* call, const char* file, unsigned int line )
{
    if( res != OPTIX_SUCCESS )
    {
        std::stringstream ss;
        ss << "Optix call '" << call << "' failed: " << file << ':' << line << ")\n";
        throw Exception( res, ss.str().c_str() );
    }
}

inline void optixCheckLog( OptixResult  res,
                           const char*  log,
                           size_t       sizeof_log,
                           size_t       sizeof_log_returned,
                           const char*  call,
                           const char*  file,
                           unsigned int line )
{
    if( res != OPTIX_SUCCESS )
    {
        std::stringstream ss;
        ss << "Optix call '" << call << "' failed: " << file << ':' << line << ")\nLog:\n"
           << log << ( sizeof_log_returned > sizeof_log ? "<TRUNCATED>" : "" ) << '\n';
        throw Exception( res, ss.str().c_str() );
    }
}

inline void optixCheckNoThrow( OptixResult res, const char* call, const char* file, unsigned int line ) noexcept
{
    if( res != OPTIX_SUCCESS )
    {
        try
        {
            std::cerr << "Optix call '" << call << "' failed: " << file << ':'
                      << line << ")\n";
        }
        catch( ... )
        {
        }
        std::terminate();
    }
}

inline void cudaCheck( cudaError_t error, const char* call, const char* file, unsigned int line )
{
    if( error != cudaSuccess )
    {
        std::stringstream ss;
        ss << "CUDA call (" << call << " ) failed with error: '"
           << cudaGetErrorString( error ) << "' (" << file << ":" << line << ")\n";
        throw Exception( ss.str().c_str() );
    }
}

inline void cudaSyncCheck( const char* file, unsigned int line )
{
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if( error != cudaSuccess )
    {
        std::stringstream ss;
        ss << "CUDA error on synchronize with error '"
           << cudaGetErrorString( error ) << "' (" << file << ":" << line << ")\n";
        throw Exception( ss.str().c_str() );
    }
}

inline void cudaCheckNoThrow( cudaError_t error, const char* call, const char* file, unsigned int line ) noexcept
{
    if( error != cudaSuccess )
    {
        try
        {
            std::cerr << "CUDA call (" << call << " ) failed with error: '"
                      << cudaGetErrorString( error ) << "' (" << file << ":"
                      << line << ")\n";
        }
        catch( ... )
        {
        }
        std::terminate();
    }
}

inline void assertCond( bool result, const char* cond, const char* file, unsigned int line )
{
    if( !result )
    {
        std::stringstream ss;
        ss << file << " (" << line << "): " << cond;
        throw Exception( ss.str().c_str() );
    }
}

inline void assertCondMsg( bool               result,
                           const char*        cond,
                           const std::string& msg,
                           const char*        file,
                           unsigned int       line )
{
    if( !result )
    {
        std::stringstream ss;
        ss << msg << ": " << file << " (" << line << "): " << cond;
        throw Exception( ss.str().c_str() );
    }
}

[[noreturn]] inline void assertFailMsg( const std::string& msg, const char* file, unsigned int line )
{
    std::stringstream ss;
    ss << msg << ": " << file << " (" << line << ')';
    throw Exception( ss.str().c_str() );
}

inline const char* getGLErrorString( GLenum error )
{
    switch( error )
    {
        case GL_NO_ERROR:
            return "No error";
        case GL_INVALID_ENUM:
            return "Invalid enum";
        case GL_INVALID_VALUE:
            return "Invalid value";
        case GL_INVALID_OPERATION:
            return "Invalid operation";
        //case GL_STACK_OVERFLOW:      return "Stack overflow";
        //case GL_STACK_UNDERFLOW:     return "Stack underflow";
        case GL_OUT_OF_MEMORY:
            return "Out of memory";
        //case GL_TABLE_TOO_LARGE:     return "Table too large";
        default:
            return "Unknown GL error";
    }
}

inline void glCheck( const char* call, const char* file, unsigned int line )
{
    GLenum err = glGetError();
    if( err != GL_NO_ERROR )
    {
        std::stringstream ss;
        ss << "GL error " << getGLErrorString( err ) << " at " << file << "("
           << line << "): " << call << '\n';
        std::cerr << ss.str() << std::endl;
        throw Exception( ss.str().c_str() );
    }
}

inline void glCheckErrors( const char* file, unsigned int line )
{
    GLenum err = glGetError();
    if( err != GL_NO_ERROR )
    {
        std::stringstream ss;
        ss << "GL error " << getGLErrorString( err ) << " at " << file << "("
           << line << ")";
        std::cerr << ss.str() << std::endl;
        throw Exception( ss.str().c_str() );
    }
}

inline void checkGLError()
{
    GLenum err = glGetError();
    if( err != GL_NO_ERROR )
    {
        std::ostringstream oss;
        do
        {
            oss << "GL error: " << getGLErrorString( err ) << '\n';
            err = glGetError();
        } while( err != GL_NO_ERROR );

        throw Exception( oss.str().c_str() );
    }
}

}  // end namespace sutil
