
/**
 * Copyright (c) 2020-2021 CutDigital Ltd.
 * All rights reserved.
 * 
 * NOTE: This file is licensed under GPL-3.0-or-later (default). 
 * A commercial license can be purchased from CutDigital Ltd. 
 *  
 * License details:
 * 
 * (A)  GNU General Public License ("GPL"); a copy of which you should have 
 *      recieved with this file.
 * 	    - see also: <http://www.gnu.org/licenses/>
 * 
 * (B)  Commercial license.
 *      - email: contact@cut-digital.com
 * 
 * The commercial license option is for users that wish to use MCUT in 
 * their products for comercial purposes but do not wish to release their 
 * software products under the GPL license. 
 * 
 */

/**
 * @file mcut.h
 * @author Floyd M. Chitalu
 * @date 1 Jan 2021
 * 
 * @brief File containing the MCUT applications programming interface (API).
 * 
 * NOTE: This header file defines all the functionality and accessible features of MCUT.
 * The interface is a standard C API.  
 * 
 */

#ifndef MCUT_API_H_
#define MCUT_API_H_

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#include "platform.h"

// Macro to encode MCUT version
#define MC_MAKE_VERSION(major, minor, patch) \
    (((major) << 22) | ((minor) << 12) | (patch))

// MCUT 1.0 version number
#define MC_API_VERSION_1_0 MC_MAKE_VERSION(1, 0, 0) // Patch version should always be set to 0

// Macro to decode MCUT version (MAJOR) from MC_HEADER_VERSION_COMPLETE
#define MC_VERSION_MAJOR(version) ((uint32_t)(version) >> 22)

// Macro to decode MCUT version (MINOR) from MC_HEADER_VERSION_COMPLETE
#define MC_VERSION_MINOR(version) (((uint32_t)(version) >> 12) & 0x3ff)

// Macro to decode MCUT version (PATCH) from MC_HEADER_VERSION_COMPLETE
#define MC_VERSION_PATCH(version) ((uint32_t)(version)&0xfff)

// Version of this file
#define MC_HEADER_VERSION 100

// Complete version of this file
#define MC_HEADER_VERSION_COMPLETE MC_MAKE_VERSION(1, 0, MC_HEADER_VERSION)

// Constant value assigned to null variables and parameters
#define MC_NULL_HANDLE 0

// Helper-macro to define opaque handles
#define MC_DEFINE_HANDLE(object) typedef struct object##_T* object;

#define MC_UNDEFINED_VALUE UINT32_MAX

/**
 * @brief Connected component handle.
 *
 * Opaque type referencing a connected component which the client must use to access mesh data after a dispatch call.
 */
typedef struct McConnectedComponent_T* McConnectedComponent;

/**
 * @brief Context handle.
 *
 * Opaque type referencing a working state (e.g. independent thread) which the client must use to initialise, dispatch, and access data.
 */
typedef struct McContext_T* McContext;

/**
 * @brief Bitfield type.
 *
 * Integral type representing a 32-bit bitfield for storing parameter values.
 */
typedef uint32_t McFlags;

/**
 * @brief Boolean type.
 *
 * Integral type representing a boolean value (MC_TRUE or MC_FALSE).
 */
typedef uint32_t McBool;

/**
 * @brief Boolean constant for "true".
 *
 * Integral constant representing a boolean value evaluating to true.
 */
#define MC_TRUE (1)

/**
 * @brief Boolean constant for "false".
 *
 * Integral constant representing a boolean value evaluating to false.
 */
#define MC_FALSE (0)

/**
 * \enum McResult
 * @brief API return codes
 *
 * This enum structure defines the possible return values of API functions (integer). The values identify whether a function executed successfully or returned with an error.
 */
typedef enum McResult {
    MC_NO_ERROR = 0, /**< The function was successfully executed. */
    MC_INVALID_OPERATION = -(1 << 1), /**< An internal operation could not be executed successively. */
    MC_INVALID_VALUE = -(1 << 2), /**< An invalid value has been passed to the API. */
    MC_OUT_OF_MEMORY = -(1 << 3), /**< Memory allocation operation cannot allocate memory. */
    MC_RESULT_MAX_ENUM = 0xFFFFFFFF /**< Wildcard (match all) . */
} McResult;

/**
 * \enum McConnectedComponentType
 * @brief The possible types of connected components.
 *
 * This enum structure defines the possible types of connected components which can be queried from the API after a dispatch call. 
 */
typedef enum McConnectedComponentType {
    MC_CONNECTED_COMPONENT_TYPE_FRAGMENT = (1 << 0), /**< A connected component that is originates from the source-mesh. */
    MC_CONNECTED_COMPONENT_TYPE_PATCH = (1 << 2), /**< A connected component that is originates from the cut-mesh. */
    MC_CONNECTED_COMPONENT_TYPE_SEAM = (1 << 3), /**< A connected component that is similer to an input mesh (source-mesh or cut-mesh), but with additional edges introduced as as a result of the cut (the intersection contour/curve). */
    MC_CONNECTED_COMPONENT_TYPE_INPUT = (1 << 4), /**< A connected component that is copy of an input mesh (source-mesh or cut-mesh). Such a connected component may contain new faces and vertices, which will happen if MCUT internally performs polygon partitioning. Polygon partitioning occurs on an input mesh which intersects the other without severing at least one edge. An example is splitting a tetrahedron (source-mesh) in two parts using one large triangle (cut-mesh): in this case, the large triangle would be partitioned into two faces to ensure that at least one of this cut-mesh are severed by the tetrahedron. */
    MC_CONNECTED_COMPONENT_TYPE_ALL = 0xFFFFFFFF /**< Wildcard (match all) . */
} McConnectedComponentType;

/**
 * \enum McFragmentLocation
 * @brief The possible geometrical locations of a fragment connected component with-respect-to the cut-mesh.
 *
 * This enum structure defines the possible locations where a fragment connected component can be relative to the cut-mesh. Note that the labels of 'above' or 'below' here are defined with-respect-to the winding-order (and hence, normal orientation) of the cut-mesh. 
 */
typedef enum McFragmentLocation {
    MC_FRAGMENT_LOCATION_ABOVE = 1 << 0, /**< Fragment is located above the cut-mesh. */
    MC_FRAGMENT_LOCATION_BELOW = 1 << 1, /**< Fragment is located below the cut-mesh. */
    MC_FRAGMENT_LOCATION_UNDEFINED = 1 << 2, /**< Fragment is located neither above nor below the cut-mesh. That is, it was produced due to a partial cut intersection. */
    MC_FRAGMENT_LOCATION_ALL = 0xFFFFFFFF /**< Wildcard (match all) . */
} McFragmentLocation;

/**
 * \enum McFragmentSealType
 * @brief Topological configurations of a fragment connected component with-respect-to hole-filling.
 *
 * This enum structure defines the possible configurations that a fragment connected component can be in regarding the hole-filling process. Here, hole-filling refers to the stage/phase when holes produced by a cut are filled with a subset of polygons of the cut-mesh.
 */
typedef enum McFragmentSealType {
    MC_FRAGMENT_SEAL_TYPE_COMPLETE = 1 << 0, /**< Holes are completely sealed (watertight). */
    MC_FRAGMENT_SEAL_TYPE_NONE = 1 << 2, /**< Holes are not sealed (gaping hole). */
    MC_FRAGMENT_SEAL_TYPE_ALL = 0xFFFFFFFF /**< Wildcard (match all) . */
} McFragmentSealType;

/**
 * \enum McPatchLocation
 * @brief Geometrical location of a patch connected component with-respect-to the source-mesh.
 *
 * This enum structure defines the possible locations where a patch connected component can be relative to the source-mesh. Note that the labels of 'inside' or 'outside' here are defined with-respect-to the winding-order (and hence, normal orientation) of the source-mesh.
 */
typedef enum McPatchLocation {
    MC_PATCH_LOCATION_INSIDE = 1 << 0, /**< Patch is located on the interior of the source-mesh (used to seal holes). */
    MC_PATCH_LOCATION_OUTSIDE = 1 << 1, /**< Patch is located on the exterior of the source-mesh. Rather than hole-filling these patches seal from the outside so-as to extrude the cut.*/
    MC_PATCH_LOCATION_UNDEFINED = 1 << 2, /**< Patch is located neither on the interior nor exterior of the source-mesh. */
    MC_PATCH_LOCATION_ALL = 0xFFFFFFFF /**< Wildcard (match all) . */
} McPatchLocation;

/**
 * \enum McSeamOrigin
 * @brief Input mesh from which a seam connected component is derived.
 *
 * This enum structure defines the possible origins of a seam connected component, which can be either the source-mesh or the cut-mesh. 
 */
typedef enum McSeamOrigin {
    MC_SEAM_ORIGIN_SRCMESH = 1 << 0, /**< Seam connected component is from the input source-mesh. */
    MC_SEAM_ORIGIN_CUTMESH = 1 << 1, /**< Seam connected component is from the input cut-mesh. */
    MC_SEAM_ORIGIN_ALL = 0xFFFFFFFF /**< Wildcard (match all) . */
} McSeamOrigin;

/**
 * \enum McInputOrigin
 * @brief The user-provided input mesh from which an input connected component is derived.
 *
 * This enum structure defines the possible origins of an input connected component, which can be either the source-mesh or the cut-mesh.
 * Note: the number of elements (faces and vertices) in an input connected component will be the same [or greater] than the corresponding user-provided input mesh from which the respective connected component came from. The input connect component will contain more elements if MCUT detected an intersection configuration where one input mesh will create a hole in a face of the other input mesh but without severing it edges (and vice versa). 
 */
typedef enum McInputOrigin {
    MC_INPUT_ORIGIN_SRCMESH = 1 << 0, /**< Input connected component from the input source mesh.*/
    MC_INPUT_ORIGIN_CUTMESH = 1 << 1, /**< Input connected component from the input cut mesh. */
    MC_INPUT_ORIGIN_ALL = 0xFFFFFFFF /**< Wildcard (match all) . */
} McInputOrigin;

/**
 * \enum McConnectedComponentData
 * @brief Data that can be queried about a connected component.
 *
 * This enum structure defines the different types of data that are associated with a connected component and can be queried from the API after a dispatch call.
 */
typedef enum McConnectedComponentData {
    //MC_CONNECTED_COMPONENT_DATA_VERTEX_COUNT = (1 << 0), /**< Number of vertices. */
    MC_CONNECTED_COMPONENT_DATA_VERTEX_FLOAT = (1 << 1), /**< List of vertex coordinates as an array of 32 bit floating-point numbers. */
    MC_CONNECTED_COMPONENT_DATA_VERTEX_DOUBLE = (1 << 2), /**< List of vertex coordinates as an array of 64 bit floating-point numbers. */
    //MC_CONNECTED_COMPONENT_DATA_FACE_COUNT = (1 << 4), /**< Number of faces. */
    MC_CONNECTED_COMPONENT_DATA_FACE = (1 << 5), /**< List of faces as an array of indices. */
    MC_CONNECTED_COMPONENT_DATA_FACE_SIZE = (1 << 6), /**< List of face sizes (vertices per face) as an array. */
    //MC_CONNECTED_COMPONENT_DATA_EDGE_COUNT = (1 << 7), /**< Number of edges. */
    MC_CONNECTED_COMPONENT_DATA_EDGE = (1 << 8), /**< List of edges as an array of indices. */
    MC_CONNECTED_COMPONENT_DATA_TYPE = (1 << 9), /**< The type of a connected component (See also: ::McConnectedComponentType.). */
    MC_CONNECTED_COMPONENT_DATA_FRAGMENT_LOCATION = (1 << 10), /**< The location of a fragment connected component with respect to the cut mesh (See also: ::McFragmentLocation). */
    MC_CONNECTED_COMPONENT_DATA_PATCH_LOCATION = (1 << 11), /**< The location of a patch with respect to the source mesh (See also: ::McPatchLocation).*/
    MC_CONNECTED_COMPONENT_DATA_FRAGMENT_SEAL_TYPE = (1 << 12), /**< The Hole-filling configuration of a fragment connected component (See also: ::McFragmentSealType). */
    MC_CONNECTED_COMPONENT_DATA_SEAM_VERTEX = (1 << 13), /**< List of seam-vertices as an array of indices.*/
    MC_CONNECTED_COMPONENT_DATA_ORIGIN = (1 << 14), /**< The input mesh (source- or cut-mesh) from which a "seam" is derived (See also: ::McSeamedConnectedComponentOrigin). */
    MC_CONNECTED_COMPONENT_DATA_VERTEX_MAP = (1 << 15), /**< List of a subset of vertex indices from one of the input meshes (source-mesh or the cut-mesh). Each value will be the index of an input mesh vertex or MC_UNDEFINED_VALUE. This index-value corresponds to the connected component vertex at the accessed index. The value at index 0 of the queried array is the index of the vertex in the original input mesh. In order to clearly distinguish indices of the cut mesh from those of the source mesh, this index value corresponds to a cut mesh vertex index if it is great-than-or-equal-to the number of source-mesh vertices. Intersection points are mapped to MC_UNDEFINED_VALUE. The input mesh (i.e. source- or cut-mesh) will be deduced by the user from the type of connected component with which the information is queried. The input connected component (source-mesh or cut-mesh) that is referred to must be one stored internally by MCUT (i.e. a connected component queried from the API via ::McInputOrigin), to ensure consistency with any modification done internally by MCUT. */
    MC_CONNECTED_COMPONENT_DATA_FACE_MAP = (1 << 16), /**< List a subset of face indices from one of the input meshes (source-mesh or the cut-mesh). Each value will be the index of an input mesh face. This index-value corresponds to the connected component face at the accessed index. Example: the value at index 0 of the queried array is the index of the face in the original input mesh. Note that all faces are mapped to a defined value. In order to clearly distinguish indices of the cut mesh from those of the source mesh, an input-mesh face index value corresponds to a cut-mesh vertex-index if it is great-than-or-equal-to the number of source-mesh faces. The input connected component (source-mesh or cut-mesh) that is referred to must be one stored internally by MCUT (i.e. a connected component queried from the API via ::McInputOrigin), to ensure consistency with any modification done internally by MCUT. */
    // incidence and adjacency information
    MC_CONNECTED_COMPONENT_DATA_FACE_ADJACENT_FACE = (1 << 17), /**< List of adjacent faces (their indices) per face.*/
    MC_CONNECTED_COMPONENT_DATA_FACE_ADJACENT_FACE_SIZE = (1 << 18) /**< List of adjacent-face-list sizes (number of adjacent faces per face).*/

} McConnectedComponentData;

/**
 * \enum McDebugSource
 * @brief Source of a debug log message.
 *
 * This enum structure defines the sources from which a message in a debug log may originate.
 */
typedef enum McDebugSource {
    MC_DEBUG_SOURCE_API = 1 << 0, /**< messages generated by usage of the MCUT API. */
    MC_DEBUG_SOURCE_KERNEL = 1 << 1, /**< messages generated by the cutting kernel. */
    MC_DEBUG_SOURCE_ALL = 0xFFFFFFFF /**< Wildcard (match all) . */
} McDebugSource;

/**
 * \enum McDebugType
 * @brief Type of debug messages.
 *
 * This enum structure defines the types of debug a message relating to an error. 
 */
typedef enum McDebugType {
    MC_DEBUG_TYPE_ERROR = 1 << 0, /**< Explicit error message.*/
    MC_DEBUG_TYPE_DEPRECATED_BEHAVIOR = 1 << 1, /**< Attempted use of deprecated features.*/
    MC_DEBUG_TYPE_OTHER = 1 << 2, /**< Other types of messages,.*/
    MC_DEBUG_TYPE_ALL = 0xFFFFFFFF /**< Wildcard (match all) . */
} McDebugType;

/**
 * \enum McDebugSeverity
 * @brief Severity levels of messages.
 *
 * This enum structure defines the different severities of error: low, medium or high severity messages.
 */
typedef enum McDebugSeverity {
    MC_DEBUG_SEVERITY_HIGH = 1 << 0, /**< All MCUT Errors, mesh conversion/parsing errors, or undefined behavior.*/
    MC_DEBUG_SEVERITY_MEDIUM = 1 << 1, /**< Major performance warnings, debugging warnings, or the use of deprecated functionality.*/
    MC_DEBUG_SEVERITY_LOW = 1 << 2, /**< Redundant state change, or unimportant undefined behavior.*/
    MC_DEBUG_SEVERITY_NOTIFICATION = 1 << 3, /**< Anything that isn't an error or performance issue.*/
    MC_DEBUG_SEVERITY_ALL = 0xFFFFFFFF /**< Match all (wildcard).*/
} McDebugSeverity;

/**
 * \enum McContextCreationFlags
 * @brief Context creation flags.
 *
 * This enum structure defines the flags with which a context can be created.
 */
typedef enum McContextCreationFlags {
    MC_DEBUG = (1 << 0), /**< Enable debug mode (message logging etc.).*/
} McContextCreationFlags;

/**
 * \enum McRoundingModeFlags
 * @brief Numerical rounding mode.
 *
 * This enum structure defines the supported rounding modes which are applied when computing intersections during a dispatch call.
 * The MC_ROUNDING_MODE_TO_NEAREST mode works as in the IEEE 754 standard: in case the number to be rounded lies exactly in the middle of two representable numbers, it is rounded to the one with the least significant bit set to zero
 */
typedef enum McRoundingModeFlags {
    MC_ROUNDING_MODE_TO_NEAREST = (1 << 2), /**< round to nearest (roundTiesToEven in IEEE 754-2008).*/
    MC_ROUNDING_MODE_TOWARD_ZERO = (1 << 3), /**< round toward zero (roundTowardZero in IEEE 754 - 2008).*/
    MC_ROUNDING_MODE_TOWARD_POS_INF = (1 << 4), /**< round toward plus infinity (roundTowardPositive in IEEE 754-2008).*/
    MC_ROUNDING_MODE_TOWARD_NEG_INF = (1 << 5) /**< round toward minus infinity (roundTowardNegative in IEEE 754-2008).*/
} McRoundingModeFlags;

/**
 * \enum McDispatchFlags
 * @brief Dispatch configuration flags.
 *
 * This enum structure defines the flags indicating MCUT is to interprete input data, and execute the cutting pipeline.
 */
typedef enum McDispatchFlags {
    MC_DISPATCH_VERTEX_ARRAY_FLOAT = (1 << 0), /**< Interpret the input mesh vertices as arrays of 32-bit floating-point numbers.*/
    MC_DISPATCH_VERTEX_ARRAY_DOUBLE = (1 << 1), /**< Interpret the input mesh vertices as arrays of 64-bit floating-point numbers.*/
    MC_DISPATCH_REQUIRE_THROUGH_CUTS = (1 << 2), /**< Require that all intersection paths/curves/contours partition the source-mesh into two disjoint parts. Otherwise, ::mcDispatch is a no-op. This flag enforces the requirement that only through-cuts are valid cuts i.e it disallows partial cuts. NOTE: This flag may not be used with ::MC_DISPATCH_FILTER_FRAGMENT_LOCATION_UNDEFINED.*/
    MC_DISPATCH_INCLUDE_VERTEX_MAP = (1 << 3), /**< Compute connected-component-to-input mesh vertex-id maps. See also: ::MC_CONNECTED_COMPONENT_DATA_VERTEX_MAP */
    MC_DISPATCH_INCLUDE_FACE_MAP = (1 << 4), /**< Compute connected-component-to-input mesh face-id maps. . See also: ::MC_CONNECTED_COMPONENT_DATA_FACE_MAP*/
    //
    MC_DISPATCH_FILTER_FRAGMENT_LOCATION_ABOVE = (1 << 5), /**< Compute fragments that are above the cut-mesh.*/
    MC_DISPATCH_FILTER_FRAGMENT_LOCATION_BELOW = (1 << 6), /**< Compute fragments that are below the cut-mesh.*/
    MC_DISPATCH_FILTER_FRAGMENT_LOCATION_UNDEFINED = (1 << 7), /**< Compute fragments that are partially cut i.e. neither above nor below the cut-mesh. NOTE: This flag may not be used with ::MC_DISPATCH_REQUIRE_THROUGH_CUTS. */
    //
    MC_DISPATCH_FILTER_FRAGMENT_SEALING_INSIDE = (1 << 8), /**< Compute fragments that are fully sealed (hole-filled) on the interior.   */
    MC_DISPATCH_FILTER_FRAGMENT_SEALING_OUTSIDE = (1 << 9), /**< Compute fragments that are fully sealed (hole-filled) on the exterior.  */
    //
    MC_DISPATCH_FILTER_FRAGMENT_SEALING_NONE = (1 << 10), /**< Compute fragments that are not sealed (holes not filled).*/
    //
    MC_DISPATCH_FILTER_PATCH_INSIDE = (1 << 11), /**< Compute patches on the inside of the source mesh (those used to fill holes).*/
    MC_DISPATCH_FILTER_PATCH_OUTSIDE = (1 << 12), /**< Compute patches on the outside of the source mesh.*/
    //
    MC_DISPATCH_FILTER_SEAM_SRCMESH = (1 << 13), /**< Compute the seam which is the same as the source-mesh but with new edges placed along the cut path. Note: a seam from the source-mesh will only be computed if the dispatch operation computes a complete (through) cut.*/
    MC_DISPATCH_FILTER_SEAM_CUTMESH = (1 << 14), /**< Compute the seam which is the same as the cut-mesh but with new edges placed along the cut path. Note: a seam from the cut-mesh will only be computed if the dispatch operation computes a complete (through) cut.*/
    //
    MC_DISPATCH_FILTER_ALL = ( //
        MC_DISPATCH_FILTER_FRAGMENT_LOCATION_ABOVE | //
        MC_DISPATCH_FILTER_FRAGMENT_LOCATION_BELOW | //
        MC_DISPATCH_FILTER_FRAGMENT_LOCATION_UNDEFINED | //
        MC_DISPATCH_FILTER_FRAGMENT_SEALING_INSIDE | //
        MC_DISPATCH_FILTER_FRAGMENT_SEALING_OUTSIDE | //
        MC_DISPATCH_FILTER_FRAGMENT_SEALING_NONE | //
        MC_DISPATCH_FILTER_PATCH_INSIDE | //
        MC_DISPATCH_FILTER_PATCH_OUTSIDE | //
        MC_DISPATCH_FILTER_SEAM_SRCMESH | //
        MC_DISPATCH_FILTER_SEAM_CUTMESH), /**< Keep all connected components resulting from the dispatched cut. */
        /** 
         * Allow MCUT to perturb the cut-mesh if the inputs are not in general position. 
         * 
         * MCUT is formulated for inputs in general position. Here the notion of general position is defined with
        respect to the orientation predicate (as evaluated on the intersecting polygons). Thus, a set of points 
        is in general position if no three points are collinear and also no four points are coplanar.

        MCUT uses the "GENERAL_POSITION_VIOLATION" flag to inform of when to use perturbation (of the
        cut-mesh) so as to bring the input into general position. In such cases, the idea is to solve the cutting
        problem not on the given input, but on a nearby input. The nearby input is obtained by perturbing the given
        input. The perturbed input will then be in general position and, since it is near the original input,
        the result for the perturbed input will hopefully still be useful.  This is justified by the fact that
        the task of MCUT is not to decide whether the input is in general position but rather to make perturbation
        on the input (if) necessary within the available precision of the computing device. */
    MC_DISPATCH_ENFORCE_GENERAL_POSITION = (1 << 15) 
} McDispatchFlags;

/**
 * \enum McQueryFlags
 * @brief Flags for querying fixed API state.
 *
 * This enum structure defines the flags which are used to query for specific information about the state of the API and/or a given context. 
 */
typedef enum McQueryFlags {
    MC_CONTEXT_FLAGS = 1 << 0, /**< Flags used to create a context.*/
    MC_DONT_CARE = 1 << 1, /**< wildcard.*/
    MC_DEFAULT_PRECISION = 1 << 2, /**< Default number of bits used to represent the significand of a floating-point number.*/
    MC_DEFAULT_ROUNDING_MODE = 1 << 3, /**< Default way to round the result of a floating-point operation.*/
    MC_PRECISION_MAX = 1 << 4, /**< Maximum value for precision bits.*/
    MC_PRECISION_MIN = 1 << 5, /**< Minimum value for precision bits.*/
} McQueryFlags;

/**
 *  
 * @brief Debug callback function signature type.
 *
 * The callback function should have this prototype (in C), or be otherwise compatible with such a prototype.
 */
typedef void (*pfn_mcDebugOutput_CALLBACK)(
    McDebugSource source,
    McDebugType type,
    unsigned int id,
    McDebugSeverity severity,
    size_t length,
    const char* message,
    const void* userParam);

/** @brief Create an MCUT context.
*
* This method creates a context object, which is a handle used by a client application to control the API state and access data.
* 
* @param [out] pContext a pointer to the allocated context handle
* @param [in] flags bitfield containing the context creation flags
*
 * An example of usage:
 * @code
 * McContext myContext = MC_NULL_HANDLE;
 * McResult err = mcCreateContext(&myContext, MC_NULL_HANDLE);
 * if(err != MC_NO_ERROR)
 * {
 *  // deal with error
 * }
 * @endcode
*
* @return Error code.
* 
* <b>Error codes</b> 
* - MC_NO_ERROR  
*   -# proper exit 
* - MC_INVALID_VALUE 
*   -# \p pContext is NULL
*   -# Failure to allocate resources
*   -# \p flags defines an invalid bitfield.
*/
extern MCAPI_ATTR McResult MCAPI_CALL mcCreateContext(
    McContext* pContext, McFlags flags);

/** @brief Set the numerical rounding mode.
*
* This function updates context state to use given rounding mode during dispatch calls. See ::McRoundingModeFlags.
*
* @param [in] context a pointer to a previous allocated context.
* @param [in] rmode The rounding mode.
*
 * An example of usage:
 * @code
 * McResult err = mcSetRoundingMode(&myContext, MC_ROUNDING_MODE_TO_NEAREST);
 * if(err != MC_NO_ERROR)
 * {
 *  // deal with error
 * }
 * @endcode
*
* @return Error code.
*
* <b>Error codes</b> 
* - MC_NO_ERROR  
*   -# proper exit 
* - MC_INVALID_VALUE 
*   -# \p pContext is NULL
*   -# \p rmode defines an invalid bitfield (e.g. more than one rounding mode).
*
* @note This function is a no-op if ARBITRARY_PRECISION_NUMBERS is not defined.
*/
extern MCAPI_ATTR McResult MCAPI_CALL mcSetRoundingMode(
    McContext context,
    McFlags rmode);

/** @brief Get the numerical rounding mode.
*
* This function retrieves the rounding mode currently used the context. @see McRoundingModeFlags
*
* @param [in] context The context handle
* @param [out] pRmode The returned value for the current rounding mode
*
 * An example of usage:
 * @code
 * McFlags myRoundingMode = 0;
 * McResult err = mcGetRoundingMode(&myContext, &myRoundingMode);
 * if(err != MC_NO_ERROR)
 * {
 *  // deal with error
 * }
 * @endcode
*
* @return Error code.
*
* <b>Error codes</b> 
* - MC_NO_ERROR  
*   -# proper exit 
* - MC_INVALID_VALUE 
*   -# \p pContext is NULL or \p pContext is not an existing context.
*   -# \p pRmode is NULL.
*/
extern MCAPI_ATTR McResult MCAPI_CALL mcGetRoundingMode(
    McContext context,
    McFlags* pRmode);

/** @brief Set the precision bits.
*
* This function sets the default precision to be exactly prec bits, where prec can be any integer between MC_PRECISION_MAX and MC_PRECISION_MIN. The precision of a variable means the number of bits used to store its significand. The default precision is set to 53 bits initially if ARBITRARY_PRECISION_NUMBERS is defined. Otherwise, the default precision is set to "sizeof(long double) * 8" bits.
*
* @param [in] context a pointer to the allocated context handle
* @param [in] prec The number precision bits
*
*
 * An example of usage:
 * @code
 * uint64_t prec = 64;
 * McResult err = mcSetPrecision(&myContext, prec);
 * if(err != MC_NO_ERROR)
 * {
 *  // deal with error
 * }
 * @endcode
*
* @return Error code.
*
* <b>Error codes</b> 
* - MC_NO_ERROR  
*   -# proper exit 
* - MC_INVALID_VALUE 
*   -# \p pContext is NULL or \p pContext is not an existing context.
*   -# \p prec is not an between #MC_PRECISION_MAX and #MC_PRECISION_MIN (See ::McQueryFlags).
*
* @note This function is a no-op if ARBITRARY_PRECISION_NUMBERS is not defined. Do not attempt to set the precision to any value near #MC_PRECISION_MAX, otherwise mcut will abort due to an assertion failure. Moreover, you may reach some memory limit on your platform, in which case the program may abort, crash or have undefined behavior (depending on your C implementation).

*/
extern MCAPI_ATTR McResult MCAPI_CALL mcSetPrecision(
    McContext context,
    uint64_t prec);

/** @brief Get the number of precision bits.
*
* This function retrieves the number of precision bits currently used by the context.
*
* @param [in] context The context handle
* @param [out] pPrec The number of precision bits
*
 * An example of usage:
 * @code
 * uint64_t myPrecisionBits = 0;
 * McResult err = mcGetPrecision(&myContext, &myPrecisionBits);
 * if(err != MC_NO_ERROR)
 * {
 *  // deal with error
 * }
 * @endcode
*
* @return Error code.
*
* <b>Error codes</b> 
* - MC_NO_ERROR  
*   -# proper exit 
* - MC_INVALID_VALUE 
*   -# \p pContext is NULL or \p pContext is not an existing context.
*   -# \p pPrec is NULL.
*
*/
extern MCAPI_ATTR McResult MCAPI_CALL mcGetPrecision(
    McContext context,
    uint64_t* pPrec);

/** @brief Specify a callback to receive debugging messages from the MCUT library.
*
* ::mcDebugMessageCallback sets the current debug output callback function to the function whose address is
* given in callback.
*
* This function is defined to have the same calling convention as the MCUT API functions. In most cases
* this is defined as MCAPI_ATTR, although it will vary depending on platform, language and compiler.
*
* Each time a debug message is generated the debug callback function will be invoked with source, type,
* and severity associated with the message, and length set to the length of debug message whose
* character string is in the array pointed to by message userParam will be set to the value passed in
* the userParam parameter to the most recent call to mcDebugMessageCallback.
*
* @param[in] context The context handle that was created by a previous call to mcCreateContext.
* @param[in] cb The address of a callback function that will be called when a debug message is generated. 
* @param[in] userParam A user supplied pointer that will be passed on each invocation of callback.
*
 * An example of usage:
 * @code
 * // define my callback (with type pfn_mcDebugOutput_CALLBACK)
 * void mcDebugOutput(McDebugSource source,   McDebugType type, unsigned int id, McDebugSeverity severity,size_t length, const char* message,const void* userParam)
 * {
 *  // do stuff
 * }
 * 
 * // ...
 * 
 * void* someData = NULL;
 * McResult err = mcDebugMessageCallback(myContext, mcDebugOutput, someData);
 * if(err != MC_NO_ERROR)
 * {
 *  // deal with error
 * }
 * @endcode
 * 
* @return Error code.
*
* <b>Error codes</b> 
* - MC_NO_ERROR  
*   -# proper exit 
* - MC_INVALID_VALUE 
*   -# \p pContext is NULL or \p pContext is not an existing context.
*   -# \p cb is NULL.
*
*/
extern MCAPI_ATTR McResult MCAPI_CALL mcDebugMessageCallback(
    McContext context,
    pfn_mcDebugOutput_CALLBACK cb,
    const void* userParam);

/**
* Control the reporting of debug messages in a debug context.
*
* @param[in] context The context handle that was created by a previous call to @see mcCreateContext.
* @param[in] source The source of debug messages to enable or disable.
* @param[in] type The type of debug messages to enable or disable.
* @param[in] severity The severity of debug messages to enable or disable.
* @param[in] enabled A Boolean flag determining whether the selected messages should be enabled or disabled.
*
* ::mcDebugMessageControl controls the reporting of debug messages generated by a debug context. The parameters 
* source, type and severity form a filter to select messages from the pool of potential messages generated by 
* the MCUT library.
*
* \p source may be #MC_DEBUG_SOURCE_API, #MC_DEBUG_SOURCE_KERNEL to select messages 
* generated by usage of the MCUT API, the MCUT kernel or by the input, respectively. It may also take the 
* value #MC_DEBUG_SOURCE_ALL. If source is not #MC_DEBUG_SOURCE_ALL then only messages whose source matches 
* source will be referenced.
*
* \p type may be one of #MC_DEBUG_TYPE_ERROR, #MC_DEBUG_TYPE_DEPRECATED_BEHAVIOR, or #MC_DEBUG_TYPE_OTHER to indicate 
* the type of messages describing MCUT errors, attempted use of deprecated features, and other types of messages, 
* respectively. It may also take the value #MC_DONT_CARE. If type is not #MC_DEBUG_TYPE_ALL then only messages whose 
* type matches type will be referenced.
*
* \p severity may be one of #MC_DEBUG_SEVERITY_LOW, #MC_DEBUG_SEVERITY_MEDIUM, or #MC_DEBUG_SEVERITY_HIGH to 
* select messages of low, medium or high severity messages or to #MC_DEBUG_SEVERITY_NOTIFICATION for notifications. 
* It may also take the value #MC_DEBUG_SEVERITY_ALL. If severity is not #MC_DEBUG_SEVERITY_ALL then only 
* messages whose severity matches severity will be referenced.
*
* If \p enabled is true then messages that match the filter formed by source, type and severity are enabled. 
* Otherwise, those messages are disabled.
*
*
 * An example of usage:
 * @code
 * // ... typically after setting debug callback with ::mcDebugMessageCallback
 * McResult err = mcDebugMessageControl(myContext, MC_DEBUG_SOURCE_ALL, MC_DEBUG_TYPE_ALL, MC_DEBUG_SEVERITY_ALL, true);
 * if(err != MC_NO_ERROR)
 * {
 *  // deal with error
 * }
 * @endcode
*
* @return Error code.
*
* <b>Error codes</b> 
* - MC_NO_ERROR  
*   -# proper exit 
* - MC_INVALID_VALUE 
*   -# \p pContext is NULL or \p pContext is not an existing context.
*   -# \p source is not a value define in ::McDebugSource.
*   -# \p type is not a value define in ::McDebugType.
*   -# \p severity is not a value define in ::McDebugSeverity.
*/
extern MCAPI_ATTR McResult MCAPI_CALL mcDebugMessageControl(
    McContext context,
    McDebugSource source,
    McDebugType type,
    McDebugSeverity severity,
    bool enabled);

/**
* @brief Execute a cutting operation with two meshes - the source mesh, and the cut mesh.
*
* @param[in] context The context handle that was created by a previous call to ::mcCreateContext.
* @param[in] flags The flags indicating how to interprete input data and configure the execution.
* @param[in] pSrcMeshVertices The vertices (x,y,z) of the source mesh.
* @param[in] pSrcMeshFaceIndices The indices of the faces (polygons) in the source mesh.
* @param[in] pSrcMeshFaceSizes The sizes (in terms of vertex indices) of the faces in the source mesh.
* @param[in] numSrcMeshVertices The number of vertices in the source mesh.
* @param[in] numSrcMeshFaces The number of faces in the source mesh.
* @param[in] pCutMeshVertices The vertices (x,y,z) of the cut mesh.
* @param[in] pCutMeshFaceIndices The indices of the faces (polygons) in the cut mesh.
* @param[in] pCutMeshFaceSizes The sizes (in terms of vertex indices) of the faces in the cut mesh.
* @param[in] numCutMeshVertices The number of vertices in the cut mesh.
* @param[in] numCutMeshFaces The number of faces in the cut mesh.
*
* This function specifies the two mesh objects to operate on. The 'source mesh' is the mesh to be cut 
* (i.e. partitioned) along intersection paths prescribed by the 'cut mesh'. 
* 
* Numerical operations are performed only to evaluate polygon intersection points. The rest of 
* the function pipeline resolves the combinatorial structure of the underlying meshes using halfedge 
* connectivity. These numerical operations are represented exact predicates which makes the routine
* also robust to floating-point error.
*
* An example of usage:
* @code
*  McResult err = mcDispatch(
*        myContext,
*        // parse vertex arrays as 32 bit vertex coordinates (float*)
*        MC_DISPATCH_VERTEX_ARRAY_FLOAT,
*        // source mesh data
*        pSrcMeshVertices,
*        pSrcMeshFaceIndices,
*        pSrcMeshFaceSizes,
*        numSrcMeshVertices,
 *       numSrcMeshFaces,
*        // cut mesh data
*        pCutMeshVertices,
*        pCutMeshFaceIndices,
*        pCutMeshFaceSizes,
*        numCutMeshVertices,
*        numCutMeshFaces);
 * if(err != MC_NO_ERROR)
 * {
 *  // deal with error
 * }
 * @endcode
* 
* @return Error code.
*
* <b>Error codes</b> 
* - ::MC_NO_ERROR  
*   -# proper exit 
* - ::MC_INVALID_VALUE 
*   -# \p pContext is NULL or \p pContext is not an existing context.
*   -# \p flags contains an invalid value.
*   -# A vertex index in \p pSrcMeshFaceIndices or \p pCutMeshFaceIndices is out of bounds.
*   -# Invalid face/polygon definition (vertex list) implying non-manifold mesh \p pSrcMeshFaceIndices or \p pCutMeshFaceIndices is out of bounds.
*   -# The MC_DISPATCH_VERTEX_ARRAY_... value has not been specified in \p flags
*   -# An input mesh contains multiple connected components.
*   -# \p pSrcMeshVertices is NULL.
*   -# \p pSrcMeshFaceIndices is NULL.
*   -# \p pSrcMeshFaceSizes is NULL.
*   -# \p numSrcMeshVertices is less than three.
*   -# \p numSrcMeshFaces is less than one.
*   -# \p pCutMeshVertices is NULL.
*   -# \p pCutMeshFaceIndices is NULL.
*   -# \p pCutMeshFaceSizes is NULL.
*   -# \p numCutMeshVertices is less than three.
*   -# \p numCutMeshFaces is less than one.
*   -# ::MC_DISPATCH_ENFORCE_GENERAL_POSITION is not set and: 1) Found two intersecting edges between the source-mesh and the cut-mesh and/or 2) An intersection test between a face and an edge failed because an edge vertex only touches (but does not penetrate) the face, and/or 3) One or more source-mesh vertices are colocated with one or more cut-mesh vertices.
* - ::MC_OUT_OF_MEMORY
*   -# Insufficient memory to perform operation.
*/
extern MCAPI_ATTR McResult MCAPI_CALL mcDispatch(
    McContext context,
    McFlags flags,
    const void* pSrcMeshVertices,
    const uint32_t* pSrcMeshFaceIndices,
    const uint32_t* pSrcMeshFaceSizes,
    uint32_t numSrcMeshVertices,
    uint32_t numSrcMeshFaces,
    const void* pCutMeshVertices,
    const uint32_t* pCutMeshFaceIndices,
    const uint32_t* pCutMeshFaceSizes,
    uint32_t numCutMeshVertices,
    uint32_t numCutMeshFaces);

/**
* @brief Return the value of a selected parameter.
*
* @param[in] context The context handle that was created by a previous call to ::mcCreateContext. 
* @param[in] info Information object being queried. ::McQueryFlags
* @param[in] bytes Size in bytes of memory pointed to by \p pMem. This size must be great than or equal to the return type size of data type queried.
* @param[out] pMem Pointer to memory where the appropriate result being queried is returned. If \p pMem is NULL, it is ignored.
* @param[out] pNumBytes returns the actual size in bytes of data being queried by info. If \p pNumBytes is NULL, it is ignored.
*
*
 * An example of usage:
 * @code
 * uint64_t numBytes = 0;
 * McFlags contextFlags;
 * McResult err =  mcGetInfo(context, MC_CONTEXT_FLAGS, 0, nullptr, &numBytes);
 * if(err != MC_NO_ERROR)
 * {
 *  // deal with error
 * }
*
 *   err = mcGetInfo(context, MC_CONTEXT_FLAGS, numBytes, &contextFlags, nullptr);
 * if(err != MC_NO_ERROR)
 * {
 *  // deal with error
 * }
 * @endcode
* @return Error code.
*
* <b>Error codes</b> 
* - MC_NO_ERROR  
*   -# proper exit 
* - MC_INVALID_VALUE 
*   -# \p pContext is NULL or \p pContext is not an existing context.
*   -# \p bytes is greater than the returned size of data type queried
*
* @note Event synchronisation is not implemented.
*/
extern MCAPI_ATTR McResult MCAPI_CALL mcGetInfo(
    const McContext context,
    McFlags info,
    uint64_t bytes,
    void* pMem,
    uint64_t* pNumBytes);

/**
* @brief Query the connected components available in a context.
* 
* This function will return an array of connected components matching the given description of flags.
*  
* @param[in] context The context handle
* @param[in] connectedComponentType The type(s) of connected component sought. See also ::McConnectedComponentType.
* @param[in] numEntries The number of ::McConnectedComponent entries that can be added to \p pConnComps. If \p pConnComps is not NULL, \p numEntries must be the number of elements in \p pConnComps.
* @param[out] pConnComps Returns a list of connected components found. The ::McConnectedComponentType values returned in \p pConnComps can be used 
* to identify a specific connected component. If \p pConnComps is NULL, this argument is ignored. The number of connected components returned 
* is the minimum of the value specified by \p numEntries or the number of connected components whose type matches \p connectedComponentType.
* @param[out] numConnComps Returns the number of connected components available that match \p connectedComponentType. If \p numConnComps is NULL, 
* this argument is ignored.
*
 * An example of usage:
 * @code
 * uint32_t numConnComps = 0;
 * McConnectedComponent* pConnComps;
 * McResult err =  err = mcGetConnectedComponents(myContext, MC_CONNECTED_COMPONENT_TYPE_ALL, 0, NULL, &numConnComps);
 * if(err != MC_NO_ERROR)
 * {
 *  // deal with error
 * }
 *
 * if (numConnComps == 0) {
 *    // ...
 * }
 *
 * pConnComps = (McConnectedComponent*)malloc(sizeof(McConnectedComponent) * numConnComps);
 *
 * err = mcGetConnectedComponents(myContext, MC_CONNECTED_COMPONENT_TYPE_ALL, numConnComps, pConnComps, NULL);
 * if(err != MC_NO_ERROR)
 * {
 *  // deal with error
 * }
 * @endcode
*
* @return Error code.
*
* <b>Error codes</b> 
* - MC_NO_ERROR  
*   -# proper exit 
* - MC_INVALID_VALUE 
*   -# \p pContext is NULL or \p pContext is not an existing context.
*   -# \p connectedComponentType is not a value in ::McConnectedComponentType.
*   -# \p numConnComps and \p pConnComps are both NULL.
*   -# \p numConnComps is zero and \p pConnComps is not NULL.
*/
extern MCAPI_ATTR McResult MCAPI_CALL mcGetConnectedComponents(
    const McContext context,
    const McConnectedComponentType connectedComponentType,
    const uint32_t numEntries,
    McConnectedComponent* pConnComps,
    uint32_t* numConnComps);

/**
* @brief Query specific information about a connected component.
*
* @param[in] context The context handle that was created by a previous call to ::mcCreateContext. 
* @param[in] connCompId A connected component returned by ::mcGetConnectedComponents whose data is to be read.
* @param[in] flags An enumeration constant that identifies the connected component information being queried.
* @param[in] bytes Specifies the size in bytes of memory pointed to by \p flags.
* @param[out] pMem A pointer to memory location where appropriate values for a given \p flags will be returned. If \p pMem is NULL, it is ignored.
* @param[out] pNumBytes Returns the actual size in bytes of data being queried by \p flags. If \p pNumBytes is NULL, it is ignored.
*
* The connected component queries described in the ::McConnectedComponentInfo should return the same information for a connected component returned by ::mcGetConnectedComponents.
*
 * An example of usage:
 * @code
 * uint64_t numBytes = 0;
 * McResult err = mcGetConnectedComponentData(myContext,  connCompId, MC_CONNECTED_COMPONENT_DATA_VERTEX_DOUBLE, 0, NULL, &numBytes);
 * if(err != MC_NO_ERROR)
 * {
 *  // deal with error
 * }
 * 
 * double* pVertices = (double*)malloc(numBytes);
 *
 * err = mcGetConnectedComponentData(context, connCompId, MC_CONNECTED_COMPONENT_DATA_VERTEX_DOUBLE, numBytes, (void*)pVertices, NULL);
 * if(err != MC_NO_ERROR)
 * {
 *  // deal with error
 * }
 * @endcode
 * 
* @return Error code.
*
* <b>Error codes</b> 
* - MC_NO_ERROR  
*   -# proper exit 
* - MC_INVALID_VALUE 
*   -# \p pContext is NULL or \p pContext is not an existing context.
*   -# \p connectedComponentType is not a value in ::McConnectedComponentType.
*   -# \p pMem and \p pNumBytes are both NULL (or not NULL).
*   -# \p bytes is zero and \p pMem is not NULL.
*   -# \p flag is MC_CONNECTED_COMPONENT_DATA_VERTEX_MAP when \p context dispatch flags did not include flag MC_DISPATCH_INCLUDE_VERTEX_MAP
*   -# \p flag is MC_CONNECTED_COMPONENT_DATA_FACE_MAP when \p context dispatch flags did not include flag MC_DISPATCH_INCLUDE_FACE_MAP
*/
extern MCAPI_ATTR McResult MCAPI_CALL mcGetConnectedComponentData(
    const McContext context,
    const McConnectedComponent connCompId,
    McFlags flags,
    uint64_t bytes,
    void* pMem,
    uint64_t* pNumBytes);

/**
* @brief To release the memory of a connected component, call this function.
*
* If \p numConnComps is zero and \p pConnComps is NULL, the memory of all connected components associated with the context is freed.
*
* @param[in] context The context handle that was created by a previous call to ::mcCreateContext.
* @param[in] numConnComps Number of connected components in \p pConnComps whose memory to release.
* @param[in] pConnComps The connected components whose memory will be released.
*
 * An example of usage:
 * @code
 * McResult err = mcReleaseConnectedComponents(myContext, pConnComps, numConnComps);
 * // OR (delete all connected components in context) 
 * //McResult err = mcReleaseConnectedComponents(myContext, NULL, 0);
 * if(err != MC_NO_ERROR)
 * {
 *  // deal with error
 * }
 * @endcode
*
* @return Error code.
*
* <b>Error codes</b> 
* - MC_NO_ERROR  
*   -# proper exit 
* - MC_INVALID_VALUE 
*   -# \p pContext is NULL or \p pContext is not an existing context.
*   -# \p numConnComps is zero and \p pConnComps is not NULL (and vice versa).
* 
*/
extern MCAPI_ATTR McResult MCAPI_CALL mcReleaseConnectedComponents(
    const McContext context,
    uint32_t numConnComps,
    const McConnectedComponent* pConnComps);

/**
* @brief To release the memory of a context, call this function.
*
* This function ensures that all the state attached to context (such as unreleased connected components, and threads) are released, and the memory is deleted.

* @param[in] context The context handle that was created by a previous call to ::mcCreateContext. 
*
*
 * An example of usage:
 * @code
 * McResult err = mcReleaseContext(myContext);
 * if(err != MC_NO_ERROR)
 * {
 *  // deal with error
 * }
 * @endcode
*
* @return Error code.
*
* <b>Error codes</b> 
* - MC_NO_ERROR  
*   -# proper exit 
* - MC_INVALID_VALUE 
*   -# \p pContext is NULL or \p pContext is not an existing context.
*/
extern MCAPI_ATTR McResult MCAPI_CALL mcReleaseContext(
    McContext context);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // #ifndef MCUT_API_H_
