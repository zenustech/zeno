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

#include <optix.h>

#include <glad/glad.h>

#include <stdexcept>
#include <sstream>
#include <iostream>

//------------------------------------------------------------------------------
//
// GL error-checking
//
//------------------------------------------------------------------------------

#define DO_GL_CHECK
#ifdef DO_GL_CHECK
#    define GL_CHECK( call )                                                   \
        do                                                                     \
        {                                                                      \
            call;                                                              \
            GLenum err = glGetError();                                         \
            if( err != GL_NO_ERROR )                                           \
            {                                                                  \
                std::stringstream ss;                                          \
                ss << "GL error " <<  sutil::getGLErrorString( err ) << " at " \
                   << __FILE__  << "(" <<  __LINE__  << "): " << #call         \
                   << std::endl;                                               \
                std::cerr << ss.str() << std::endl;                            \
                throw sutil::Exception( ss.str().c_str() );                    \
            }                                                                  \
        }                                                                      \
        while (0)


#    define GL_CHECK_ERRORS( )                                                 \
        do                                                                     \
        {                                                                      \
            GLenum err = glGetError();                                         \
            if( err != GL_NO_ERROR )                                           \
            {                                                                  \
                std::stringstream ss;                                          \
                ss << "GL error " <<  sutil::getGLErrorString( err ) << " at " \
                   << __FILE__  << "(" <<  __LINE__  << ")";                   \
                std::cerr << ss.str() << std::endl;                            \
                throw sutil::Exception( ss.str().c_str() );                    \
            }                                                                  \
        }                                                                      \
        while (0)

#else
#    define GL_CHECK( call )   do { call; } while(0)
#    define GL_CHECK_ERRORS( ) do { ;     } while(0)
#endif


//------------------------------------------------------------------------------
//
// OptiX error-checking
//
//------------------------------------------------------------------------------

#define OPTIX_CHECK( call )                                                    \
    do                                                                         \
    {                                                                          \
        OptixResult res = call;                                                \
        if( res != OPTIX_SUCCESS )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "Optix call '" << #call << "' failed: " __FILE__ ":"         \
               << __LINE__ << ")\n";                                           \
            throw sutil::Exception( res, ss.str().c_str() );                   \
        }                                                                      \
    } while( 0 )


#define OPTIX_CHECK_LOG( call )                                                \
    do                                                                         \
    {                                                                          \
        OptixResult res = call;                                                \
        const size_t sizeof_log_returned = sizeof_log;                         \
        sizeof_log = sizeof( log ); /* reset sizeof_log for future calls */    \
        if( res != OPTIX_SUCCESS )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "Optix call '" << #call << "' failed: " __FILE__ ":"         \
               << __LINE__ << ")\nLog:\n" << log                               \
               << ( sizeof_log_returned > sizeof( log ) ? "<TRUNCATED>" : "" ) \
               << "\n";                                                        \
            throw sutil::Exception( res, ss.str().c_str() );                   \
        }                                                                      \
    } while( 0 )

// This version of the log-check macro doesn't require the user do setup
// a log buffer and size variable in the surrounding context; rather the
// macro defines a log buffer and log size variable (LOG and LOG_SIZE)
// respectively that should be passed to the message being checked.
// E.g.:
//  OPTIX_CHECK_LOG2( optixProgramGroupCreate( ..., LOG, &LOG_SIZE, ... );
//
#define OPTIX_CHECK_LOG2( call )                                               \
    do                                                                         \
    {                                                                          \
        char               LOG[400];                                           \
        size_t             LOG_SIZE = sizeof( LOG );                           \
                                                                               \
        OptixResult res = call;                                                \
        if( res != OPTIX_SUCCESS )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "Optix call '" << #call << "' failed: " __FILE__ ":"         \
               << __LINE__ << ")\nLog:\n" << LOG                               \
               << ( LOG_SIZE > sizeof( LOG ) ? "<TRUNCATED>" : "" )            \
               << "\n";                                                        \
            throw sutil::Exception( res, ss.str().c_str() );                   \
        }                                                                      \
    } while( 0 )

#define OPTIX_CHECK_NOTHROW( call )                                            \
    do                                                                         \
    {                                                                          \
        OptixResult res = call;                                                \
        if( res != OPTIX_SUCCESS )                                             \
        {                                                                      \
            std::cerr << "Optix call '" << #call                               \
                      << "' failed: " __FILE__ ":" << __LINE__ << ")\n";       \
            std::terminate();                                                  \
        }                                                                      \
    } while( 0 )

//------------------------------------------------------------------------------
//
// CUDA error-checking
//
//------------------------------------------------------------------------------

#define CUDA_CHECK( call )                                                     \
    do                                                                         \
    {                                                                          \
        cudaError_t error = call;                                              \
        if( error != cudaSuccess )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "CUDA call (" << #call << " ) failed with error: '"          \
               << cudaGetErrorString( error )                                  \
               << "' (" __FILE__ << ":" << __LINE__ << ")\n";                  \
            throw sutil::Exception( ss.str().c_str() );                        \
        }                                                                      \
    } while( 0 )


#define CUDA_SYNC_CHECK()                                                      \
    do                                                                         \
    {                                                                          \
        cudaDeviceSynchronize();                                               \
        cudaError_t error = cudaGetLastError();                                \
        if( error != cudaSuccess )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "CUDA error on synchronize with error '"                     \
               << cudaGetErrorString( error )                                  \
               << "' (" __FILE__ << ":" << __LINE__ << ")\n";                  \
            throw sutil::Exception( ss.str().c_str() );                        \
        }                                                                      \
    } while( 0 )


// A non-throwing variant for use in destructors.
// An iostream must be provided for output (e.g. std::cerr).
#define CUDA_CHECK_NOTHROW( call )                                             \
    do                                                                         \
    {                                                                          \
        cudaError_t error = (call);                                            \
        if( error != cudaSuccess )                                             \
        {                                                                      \
            std::cerr << "CUDA call (" << #call << " ) failed with error: '"  \
               << cudaGetErrorString( error )                                  \
               << "' (" __FILE__ << ":" << __LINE__ << ")\n";                  \
            std::terminate();                                                  \
        }                                                                      \
    } while( 0 )

//------------------------------------------------------------------------------
//
// Assertions
//
//------------------------------------------------------------------------------

#define SUTIL_ASSERT( cond )                                                   \
    do                                                                         \
    {                                                                          \
        if( !(cond) )                                                          \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << __FILE__ << " (" << __LINE__ << "): " << #cond;              \
            throw sutil::Exception( ss.str().c_str() );                        \
        }                                                                      \
    } while( 0 )


#define SUTIL_ASSERT_MSG( cond, msg )                                          \
    do                                                                         \
    {                                                                          \
        if( !(cond) )                                                          \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << (msg) << ": " << __FILE__ << " (" << __LINE__ << "): " << #cond ; \
            throw sutil::Exception( ss.str().c_str() ); \
        }                                                                      \
    } while( 0 )

namespace sutil
{

class Exception : public std::runtime_error
{
 public:
     Exception( const char* msg )
         : std::runtime_error( msg )
     { }

     Exception( OptixResult res, const char* msg )
         : std::runtime_error( createMessage( res, msg ).c_str() )
     { }

 private:
     std::string createMessage( OptixResult res, const char* msg )
     {
         std::ostringstream out;
         out << optixGetErrorName( res ) << ": " << msg;
         return out.str();
     }
};


inline const char* getGLErrorString( GLenum error )
{
    switch( error )
    {
        case GL_NO_ERROR:            return "No error";
        case GL_INVALID_ENUM:        return "Invalid enum";
        case GL_INVALID_VALUE:       return "Invalid value";
        case GL_INVALID_OPERATION:   return "Invalid operation";
        //case GL_STACK_OVERFLOW:      return "Stack overflow";
        //case GL_STACK_UNDERFLOW:     return "Stack underflow";
        case GL_OUT_OF_MEMORY:       return "Out of memory";
        //case GL_TABLE_TOO_LARGE:     return "Table too large";
        default:                     return "Unknown GL error";
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
            oss << "GL error: " << getGLErrorString( err ) << "\n";
            err = glGetError();
        }
        while( err != GL_NO_ERROR );

        throw Exception( oss.str().c_str() );
    }
}


} // end namespace sutil
