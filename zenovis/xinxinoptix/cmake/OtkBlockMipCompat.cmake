# OTK 0.9.1's BC mip-tail upload rounds sub-4x4 dimensions down to zero.
# CUDA exposes logical 2x2 and 1x1 levels, each backed by a complete BC block.
# Compile a corrected build-tree copy; keep the pinned SDK checkout untouched.
function(zeno_otk_block_mip_compat)
    get_target_property(_source_dir DemandLoading SOURCE_DIR)
    set(_original "${_source_dir}/src/Textures/SparseTexture.cpp")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_original}")
    file(READ "${_original}" _text)
    set(_old_pitch "copyArgs.srcPitch      = ( blockScale * levelDims.x * bitsPerPixel ) / BITS_PER_BYTE;")
    set(_old_height "copyArgs.Height        = levelDims.y / blockScale;")
    set(_old_offset "offset += ( levelDims.x * levelDims.y * bitsPerPixel ) / BITS_PER_BYTE;")
    foreach(_pattern _old_pitch _old_height _old_offset)
        string(FIND "${_text}" "${${_pattern}}" _found)
        if(_found EQUAL -1)
            message(FATAL_ERROR "OTK BC mip-tail source changed; review the compatibility adapter")
        endif()
    endforeach()
    string(REPLACE "${_old_pitch}" "copyArgs.srcPitch      = ( blockScale * ((levelDims.x + blockScale - 1) / blockScale * blockScale) * bitsPerPixel ) / BITS_PER_BYTE;" _text "${_text}")
    string(REPLACE "${_old_height}" "copyArgs.Height        = (levelDims.y + blockScale - 1) / blockScale;" _text "${_text}")
    string(REPLACE "${_old_offset}" "offset += copyArgs.WidthInBytes * copyArgs.Height;" _text "${_text}")
    string(PREPEND _text "#include \"BatchedTextureUpload.h\"\n#include \"ClusterProfiling.h\"\n")
    string(REPLACE "void SparseArray::init( const imageSource::TextureInfo& info )\n{"
        "void SparseArray::init( const imageSource::TextureInfo& info )\n{\n    zeno::cluster::TimelineRange timeline(\"Textures: sparse array init\",(uint64_t(info.width)<<32)|info.height);" _text "${_text}")
    string(REPLACE "cuMemMapArrayAsync( &mapInfo, 1, stream )" "zeno::cluster::mapTextureArray( &mapInfo, 1, stream )" _text "${_text}")
    string(REPLACE "cuMemcpy2DAsync( &copyArgs, stream )" "zeno::cluster::copyTextureTile( &copyArgs, stream )" _text "${_text}")
    set(_generated "${CMAKE_CURRENT_BINARY_DIR}/otk_bc_mips/SparseTexture.cpp")
    set(_previous "")
    if(EXISTS "${_generated}")
        file(READ "${_generated}" _previous)
    endif()
    if(NOT _previous STREQUAL _text)
        file(WRITE "${_generated}" "${_text}")
    endif()
    get_target_property(_sources DemandLoading SOURCES)
    set(_next "")
    foreach(_source IN LISTS _sources)
        if(_source STREQUAL "src/Textures/SparseTexture.cpp" OR _source STREQUAL "${_original}")
            list(APPEND _next "${_generated}")
        elseif(IS_ABSOLUTE "${_source}" OR _source MATCHES "^\\$<")
            list(APPEND _next "${_source}")
        else()
            list(APPEND _next "${_source_dir}/${_source}")
        endif()
    endforeach()
    set_property(TARGET DemandLoading PROPERTY SOURCES "${_next}")
    target_include_directories(DemandLoading PRIVATE "${_source_dir}/src/Textures")
endfunction()

function(zeno_otk_request_dispatch_compat)
    get_target_property(_source_dir DemandLoading SOURCE_DIR)
    set(_original "${_source_dir}/src/PagingSystem.cpp")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_original}")
    file(READ "${_original}" _text)
    set(_old "#include \"Util/CudaCallback.h\"")
    string(FIND "${_text}" "${_old}" _found)
    if(_found EQUAL -1)
        message(FATAL_ERROR "OTK paging callback source changed; review request dispatch adapter")
    endif()
    string(REPLACE "${_old}" "#include \"DeferredTextureCallbacks.h\"" _text "${_text}")
    set(_old "for( StagedPageList s : m_stagedPages )")
    string(FIND "${_text}" "${_old}" _found)
    if(_found EQUAL -1)
        message(FATAL_ERROR "OTK staged-page accounting changed; review eviction adapter")
    endif()
    # Counting the eviction queue must not copy every staged page deque.
    string(REPLACE "${_old}" "for( const StagedPageList& s : m_stagedPages )" _text "${_text}")
    # Keep old tiles sampleable until new demand actually needs their space.
    # SDK invalidation, the render-complete event, and physical reuse retain
    # their established order once reclamation is necessary.
    set(_old "if( m_evictionActive && getNumStagedPages() < m_options->maxStagedPages )")
    string(FIND "${_text}" "${_old}" _found)
    if(_found EQUAL -1)
        message(FATAL_ERROR "OTK invalidation policy changed; review standby adapter")
    endif()
    string(REPLACE "${_old}" "if( m_evictionActive && getNumStagedPages() < m_options->maxStagedPages\n            && (!zeno::cluster::retainTextureStandby() ||\n                (numRequestedPages && m_deviceMemoryManager->needTileBlocksFreed())) )" _text "${_text}")
    set(_old "    unsigned int numRequestedPages = pinnedRequestContext->arrayLengths[PAGE_REQUESTS_LENGTH];")
    string(REPLACE "${_old}" "    const auto pagingDiagnosticStart = std::chrono::steady_clock::now();\n    unsigned int pagingRestored=0,pagingInvalidated=0;\n${_old}" _text "${_text}")
    string(REPLACE "        if( restoreMapping( pinnedRequestContext->requestedPages[i] ) )\n        {" "        if( restoreMapping( pinnedRequestContext->requestedPages[i] ) )\n        {\n            ++pagingRestored;" _text "${_text}")
    string(REPLACE "            stageStalePages( pinnedRequestContext, m_stagedPages.back().mappings );" "            const auto before=m_pageMappingsContext->numInvalidatedPages;\n            stageStalePages( pinnedRequestContext, m_stagedPages.back().mappings );\n            pagingInvalidated=m_pageMappingsContext->numInvalidatedPages-before;" _text "${_text}")
    set(_old "    updateLruThreshold( numStalePages, pinnedRequestContext->maxStalePages, medianLruVal );")
    set(_new [=[    if( zeno::cluster::textureWorkDiagnosticsEnabled() )
    {
        unsigned int empty = 0;
        for( const auto& list : m_stagedPages ) empty += list.mappings.empty();
        zeno::cluster::noteTexturePaging( static_cast<unsigned int>(m_stagedPages.size()), empty,
            static_cast<unsigned int>(getNumStagedPages()), numRequestedPages, numStalePages, m_evictionActive,
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-pagingDiagnosticStart).count(),
            pagingRestored,pagingInvalidated,zeno::cluster::retainTextureStandby() );
    }
]=])
    string(REPLACE "${_old}" "${_new}${_old}" _text "${_text}")
    string(PREPEND _text "#include \"CompletedTextureUpload.h\"\n#include \"BatchedTextureUpload.h\"\n#include <chrono>\n")
    set(_generated "${CMAKE_CURRENT_BINARY_DIR}/otk_callbacks/PagingSystem.cpp")
    set(_previous "")
    if(EXISTS "${_generated}")
        file(READ "${_generated}" _previous)
    endif()
    if(NOT _previous STREQUAL _text)
        file(WRITE "${_generated}" "${_text}")
    endif()
    get_target_property(_sources DemandLoading SOURCES)
    list(REMOVE_ITEM _sources "src/PagingSystem.cpp" "${_original}")
    list(APPEND _sources "${_generated}")
    set(_original "${_source_dir}/src/DemandLoaderImpl.cpp")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_original}")
    file(READ "${_original}" _text)
    set(_old "DemandLoaderImpl::~DemandLoaderImpl()\n{\n    m_requestProcessor.stop();")
    string(FIND "${_text}" "${_old}" _found)
    if(_found EQUAL -1)
        message(FATAL_ERROR "OTK loader teardown changed; review deferred callback lifetime")
    endif()
    string(REPLACE "${_old}" "DemandLoaderImpl::~DemandLoaderImpl()\n{\n    CudaCallback::drain();\n    m_requestProcessor.stop();\n    zeno::cluster::releasePreparedTextureUploads(this);\n    zeno::cluster::drainTextureUploads(this);\n    zeno::cluster::forgetTextureUploadStatistics(this);" _text "${_text}")
    string(REPLACE "    zeno::cluster::forgetTextureUploadStatistics(this);" "    if(zeno::cluster::textureWorkDiagnosticsEnabled())\n        std::fprintf(stderr,\"[CLAS texture pool] backing_bytes=%llu\\n\",static_cast<unsigned long long>(getDeviceMemoryManager()->getTextureTileMemory()));\n    zeno::cluster::forgetTextureUploadStatistics(this);" _text "${_text}")
    set(_old "    while( getDeviceMemoryManager()->needTileBlocksFreed() )")
    string(FIND "${_text}" "${_old}" _found)
    if(_found EQUAL -1)
        message(FATAL_ERROR "OTK staged-tile eviction changed; review eviction budget")
    endif()
    # This loop owns the same loader mutex as render-thread processRequests().
    # Its default 8192-page reserve can otherwise cause seconds of unmapping
    # in one lock hold when the pool fills. Keep the SDK's page/event ordering
    # and allocation retry behavior, but reclaim incrementally on tile loads.
    # One individual driver call can still exceed the soft time budget.
    set(_new [=[    const auto evictionStart = std::chrono::steady_clock::now();
    unsigned int evictionCount = 0;
    while( evictionCount++ < zeno::cluster::preparedTextureReclaimPages(this) && getDeviceMemoryManager()->needTileBlocksFreed()
           && std::chrono::steady_clock::now() - evictionStart < std::chrono::milliseconds(1) )]=])
    string(REPLACE "${_old}" "${_new}" _text "${_text}")
    set(_old "    PageMapping   mapping;")
    string(FIND "${_text}" "${_old}" _found)
    if(_found EQUAL -1)
        message(FATAL_ERROR "OTK tile reclamation changed; review map batching")
    endif()
    string(REPLACE "${_old}" "${_old}\n    zeno::cluster::ScopedTextureUnmaps batchedUnmaps(this,stream);" _text "${_text}")
    string(REPLACE "            getDeviceMemoryManager()->freeTileBlock( mapping.page );" "            zeno::cluster::afterTextureUnmaps([this,mapping]{getDeviceMemoryManager()->freeTileBlock(mapping.page);});" _text "${_text}")
    set(_old "void DemandLoaderImpl::freeTransferBuffer( const TransferBufferDesc& transferBuffer, CUstream stream )\n{")
    string(FIND "${_text}" "${_old}" _found)
    if(_found EQUAL -1)
        message(FATAL_ERROR "OTK staging release changed; review map batching")
    endif()
    set(_new [=[
    if(zeno::cluster::capturingTextureTransfer()) {
        if(zeno::cluster::recycleCompletedTextureTransfers()) {
            zeno::cluster::deferTextureTransferAction([this,transferBuffer]{
                if(transferBuffer.memoryType==CU_MEMORYTYPE_HOST)
                    m_pageLoader->getPinnedMemoryPool()->free(transferBuffer.memoryBlock);
                else if(transferBuffer.memoryType==CU_MEMORYTYPE_DEVICE)
                    m_deviceTransferPool.free(transferBuffer.memoryBlock);
                else OTK_ASSERT_MSG(false,"Unknown memory type.");
                zeno::cluster::noteTextureTransferRecycle(this);
            });
        } else zeno::cluster::deferTextureTransferAction([this,transferBuffer,stream]{freeTransferBuffer(transferBuffer,stream);});
        return;
    }
]=])
    string(REPLACE "${_old}" "${_old}\n${_new}" _text "${_text}")
    set(_old "        textureRequestHandler->unmapTileResource( stream, pageId );")
    string(FIND "${_text}" "${_old}" _found)
    if(_found EQUAL -1)
        message(FATAL_ERROR "OTK texture unmapping changed; review eviction diagnostics")
    endif()
    set(_new [=[    {
        const auto evictionSubmitStart = std::chrono::steady_clock::now();
        textureRequestHandler->unmapTileResource( stream, pageId );
        zeno::cluster::noteTextureEviction( this, std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - evictionSubmitStart ).count() );
    }]=])
    string(REPLACE "${_old}" "${_new}" _text "${_text}")
    string(PREPEND _text "#include \"DeferredTextureCallbacks.h\"\n#include \"CompletedTextureUpload.h\"\n#include \"BatchedTextureUpload.h\"\n#include <chrono>\n#include <cstdio>\n")
    set(_generated "${CMAKE_CURRENT_BINARY_DIR}/otk_callbacks/DemandLoaderImpl.cpp")
    set(_previous "")
    if(EXISTS "${_generated}")
        file(READ "${_generated}" _previous)
    endif()
    if(NOT _previous STREQUAL _text)
        file(WRITE "${_generated}" "${_text}")
    endif()
    list(REMOVE_ITEM _sources "src/DemandLoaderImpl.cpp" "${_original}")
    list(APPEND _sources "${_generated}")
    # Filled mappings can be published on the render stream only after their
    # tile/sampler copy completes. A completion queue preserves IO concurrency
    # without placing every frame behind the shared upload-stream queue.
    foreach(_handler TextureRequestHandler SamplerRequestHandler)
        set(_original "${_source_dir}/src/Textures/${_handler}.cpp")
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_original}")
        file(READ "${_original}" _text)
        string(REGEX MATCHALL "m_loader->setPageTableEntry\\(" _matches "${_text}")
        list(LENGTH _matches _count)
        if((_handler STREQUAL "TextureRequestHandler" AND NOT _count EQUAL 3) OR
           (_handler STREQUAL "SamplerRequestHandler" AND NOT _count EQUAL 2))
            message(FATAL_ERROR "OTK mapping publication changed; review completed-upload adapter")
        endif()
        string(REPLACE "m_loader->setPageTableEntry(" "zeno::cluster::publishTextureUpload( stream, m_loader, " _text "${_text}")
        set(_old "void ${_handler}::loadPage( CUstream stream, unsigned int pageId, bool reloadIfResident )\n{")
        string(FIND "${_text}" "${_old}" _found)
        if(_found EQUAL -1)
            message(FATAL_ERROR "OTK upload entry changed; review dedicated texture stream")
        endif()
        # A mapping stream must not contain feedback waits for future renders.
        # All tile/tail/sampler uploads for this loader share its upload stream;
        # completed-copy events still gate render-side publication.
        string(REPLACE "${_old}" "${_old}\n    zeno::cluster::TimelineRange timeline(\"Textures: ${_handler} load\",pageId);\n    stream=zeno::cluster::textureUploadStream(m_loader,stream);" _text "${_text}")
        if(_handler STREQUAL "TextureRequestHandler")
            set(_old "    if( satisfied )\n    {\n        // Coalesce white/black tiles")
            string(FIND "${_text}" "${_old}" _found)
            if(_found EQUAL -1)
                message(FATAL_ERROR "OTK tile fill changed; review batched transfer lifetime")
            endif()
            string(REPLACE "${_old}" "    zeno::cluster::ScopedTextureTransfer batchedTransfer(m_loader,pageId,stream,useNewBlock);\n${_old}" _text "${_text}")
            set(_old "MutexArrayLock lock( m_mutex.get(), index);")
        else()
            set(_old "MutexArrayLock lock( m_mutex.get(), pageId - m_startPage );")
        endif()
        string(FIND "${_text}" "${_old}" _found)
        if(_found EQUAL -1)
            message(FATAL_ERROR "OTK request locking changed; review in-flight upload guard")
        endif()
        string(REPLACE "${_old}" "${_old}\n    if(zeno::cluster::textureUploadPending(m_loader,pageId))return;" _text "${_text}")
        if(_handler STREQUAL "TextureRequestHandler")
            set(_old "    m_loader->freeStagedTiles( stream );")
            string(REPLACE "${_old}" "    if(!zeno::cluster::usePreparedTextureUploads(m_loader) || zeno::cluster::executingPreparedTextureUpload(m_loader,pageId))\n        m_loader->freeStagedTiles( stream );" _text "${_text}")
            set(_old "void TextureRequestHandler::fillTileRequest( CUstream stream, unsigned int pageId, TileBlockHandle bh )\n{")
            set(_new [=[
    if(m_texture->getFillType()==CU_MEMORYTYPE_HOST && bh.block.isBad() &&
       zeno::cluster::usePreparedTextureUploads(m_loader) && !zeno::cluster::executingPreparedTextureUpload(m_loader,pageId)) {
        unsigned int mip,x,y;
        unpackTileIndex(m_texture->getSampler(),pageId-m_startPage,mip,x,y);
        std::vector<char> bytes(TILE_SIZE_IN_BYTES);
        if(m_texture->readTile(mip,x,y,bytes.data(),bytes.size(),stream))
            zeno::cluster::enqueuePreparedTextureUpload(m_loader,pageId,std::move(bytes),[this,pageId](CUstream uploadStream){loadPage(uploadStream,pageId,false);},(uint64_t(m_startPage)<<32)|mip);
        return;
    }
]=])
            string(FIND "${_text}" "${_old}" _found)
            if(_found EQUAL -1)
                message(FATAL_ERROR "OTK tile handler changed; review CPU preparation adapter")
            endif()
            string(REPLACE "${_old}" "${_old}\n${_new}" _text "${_text}")
            set(_old "void TextureRequestHandler::fillMipTailRequest( CUstream stream, unsigned int pageId, TileBlockHandle bh )\n{")
            set(_new [=[
    if(m_texture->getFillType()==CU_MEMORYTYPE_HOST && bh.block.isBad() &&
       zeno::cluster::usePreparedTextureUploads(m_loader) && !zeno::cluster::executingPreparedTextureUpload(m_loader,pageId)) {
        std::vector<char> bytes(m_texture->getMipTailSize());
        if(m_texture->readMipTail(bytes.data(),bytes.size(),stream))
            zeno::cluster::enqueuePreparedTextureUpload(m_loader,pageId,std::move(bytes),[this,pageId](CUstream uploadStream){loadPage(uploadStream,pageId,false);},(uint64_t(m_startPage)<<32)|0xffffffffu);
        return;
    }
]=])
            string(FIND "${_text}" "${_old}" _found)
            if(_found EQUAL -1)
                message(FATAL_ERROR "OTK mip-tail handler changed; review CPU preparation adapter")
            endif()
            string(REPLACE "${_old}" "${_old}\n${_new}" _text "${_text}")
            string(REPLACE "satisfied = m_texture->readTile(" "satisfied = zeno::cluster::copyPreparedTextureData(m_loader,pageId,reinterpret_cast<char*>(transferBuffer.memoryBlock.ptr),transferBuffer.memoryBlock.size) || m_texture->readTile(" _text "${_text}")
            string(REPLACE "satisfied = m_texture->readMipTail(" "satisfied = zeno::cluster::copyPreparedTextureData(m_loader,pageId,reinterpret_cast<char*>(transferBuffer.memoryBlock.ptr),transferBuffer.memoryBlock.size) || m_texture->readMipTail(" _text "${_text}")
        else()
            set(_old "    loadPage( stream, pageId, false );")
            string(REPLACE "${_old}" "    if(zeno::cluster::usePreparedTextureUploads(m_loader)) {\n        if(zeno::cluster::textureUploadPending(m_loader,pageId) || m_loader->getPagingSystem()->isResident(pageId))return;\n        zeno::cluster::enqueuePreparedTextureUpload(m_loader,pageId,{},[this,pageId](CUstream uploadStream){loadPage(uploadStream,pageId,false);});\n        return;\n    }\n${_old}" _text "${_text}")
        endif()
        string(PREPEND _text "#include \"CompletedTextureUpload.h\"\n#include \"BatchedTextureUpload.h\"\n#include \"ClusterProfiling.h\"\n")
        set(_generated "${CMAKE_CURRENT_BINARY_DIR}/otk_callbacks/${_handler}.cpp")
        set(_previous "")
        if(EXISTS "${_generated}")
            file(READ "${_generated}" _previous)
        endif()
        if(NOT _previous STREQUAL _text)
            file(WRITE "${_generated}" "${_text}")
        endif()
        list(REMOVE_ITEM _sources "src/Textures/${_handler}.cpp" "${_original}")
        list(APPEND _sources "${_generated}")
    endforeach()
    # Allocation and callback-side recycling use different outer SDK mutexes.
    # Protect the shared DeviceContext vectors with one lock on both paths.
    set(_original "${_source_dir}/src/Memory/DeviceMemoryManager.cpp")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_original}")
    file(READ "${_original}" _text)
    foreach(_method "DeviceContext* DeviceMemoryManager::allocateDeviceContext()" "void DeviceMemoryManager::freeDeviceContext( DeviceContext* context )")
        set(_old "${_method}\n{")
        string(FIND "${_text}" "${_old}" _found)
        if(_found EQUAL -1)
            message(FATAL_ERROR "OTK context recycling changed; review shared-context lock")
        endif()
        string(REPLACE "${_old}" "${_old}\n    std::lock_guard<std::mutex> contextLock(zenoTextureContextMutex);" _text "${_text}")
    endforeach()
    # Direct allocations without a suballocator are not tracked by MemoryPool;
    # DeviceContextImpl relies on that pool to own and free its arrays. Use a
    # small tracked heap so reset/destruction releases every context buffer.
    set(_old "m_deviceContextMemory( new DeviceAllocator(), nullptr )")
    string(FIND "${_text}" "${_old}" _found)
    if(_found EQUAL -1)
        message(FATAL_ERROR "OTK context allocation changed; review buffer ownership")
    endif()
    string(REPLACE "${_old}" "m_deviceContextMemory( new DeviceAllocator(), new HeapSuballocator(), 65536, 0, 0 )" _text "${_text}")
    string(PREPEND _text "#include <mutex>\nnamespace { std::mutex zenoTextureContextMutex; }\n")
    set(_generated "${CMAKE_CURRENT_BINARY_DIR}/otk_callbacks/DeviceMemoryManager.cpp")
    set(_previous "")
    if(EXISTS "${_generated}")
        file(READ "${_generated}" _previous)
    endif()
    if(NOT _previous STREQUAL _text)
        file(WRITE "${_generated}" "${_text}")
    endif()
    list(REMOVE_ITEM _sources "src/Memory/DeviceMemoryManager.cpp" "${_original}")
    list(APPEND _sources "${_generated}")
    # Pool shrinking must unmap sparse-array references as well as invalidating
    # software page bits. cuMemRelease alone keeps mapped physical storage alive.
    set(_original "${_source_dir}/src/DemandPageLoaderImpl.cpp")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_original}")
    file(READ "${_original}" _text)
    set(_old "return ( tileBlock.arenaId >= m_maxArenas );")
    string(FIND "${_text}" "${_old}" _found)
    if(_found EQUAL -1)
        message(FATAL_ERROR "OTK resize predicate changed; review physical sparse unmapping")
    endif()
    string(REPLACE "#include \"TicketImpl.h\"" "#include \"TicketImpl.h\"\n#include \"Textures/TextureRequestHandler.h\"\n#include \"Textures/DemandTextureImpl.h\"\n#include \"BatchedTextureUpload.h\"" _text "${_text}")
    string(REPLACE "DeviceMemoryManager* deviceMemoryManager, unsigned int maxArenas" "DeviceMemoryManager* deviceMemoryManager, PageTableManager* table, unsigned int maxArenas" _text "${_text}")
    string(REPLACE "m_maxArenas( maxArenas )" "m_maxArenas( maxArenas ), m_table( table )" _text "${_text}")
    string(REPLACE "unsigned int /*pageId*/, unsigned long long pageVal, CUstream /*stream*/" "unsigned int pageId, unsigned long long pageVal, CUstream stream" _text "${_text}")
    set(_new [=[if( tileBlock.arenaId < m_maxArenas )return false;
        // invalidatePages holds PagingSystem's mutex: unmap the immutable
        // texture resource directly, not unmapTileResource (which re-locks it).
        auto* handler=dynamic_cast<TextureRequestHandler*>(m_table->getRequestHandler(pageId));
        if(handler) {
            auto* texture=handler->getTexture();const auto& sampler=texture->getSampler();
            const auto tile=pageId-sampler.startPage;
            if(!tile)texture->unmapMipTail(stream);
            else {
                unsigned int level,x,y;unpackTileIndex(sampler,tile,level,x,y);
                texture->unmapTile(stream,level,x,y);
            }
        }
        return true;]=])
    string(REPLACE "${_old}" "${_new}" _text "${_text}")
    string(REPLACE "unsigned int m_maxArenas;" "unsigned int m_maxArenas;\n    PageTableManager* m_table;" _text "${_text}")
    string(REPLACE "ResizeTilePoolPredicate( &m_deviceMemoryManager, static_cast<unsigned int>( maxArenas ) )" "ResizeTilePoolPredicate( &m_deviceMemoryManager, m_pageTableManager.get(), static_cast<unsigned int>( maxArenas ) )" _text "${_text}")
    string(REPLACE "        invalidatePages( stream, context );\n        cuStreamSynchronize( stream );" "        {\n            zeno::cluster::ScopedTextureUnmaps batch(nullptr,stream);\n            invalidatePages( stream, context );\n        }\n        cuStreamSynchronize( stream );" _text "${_text}")
    set(_generated "${CMAKE_CURRENT_BINARY_DIR}/otk_callbacks/DemandPageLoaderImpl.cpp")
    _zeno_otk_write_changed("${_generated}" "${_text}")
    list(REMOVE_ITEM _sources "src/DemandPageLoaderImpl.cpp" "${_original}")
    list(APPEND _sources "${_generated}")
    set_property(TARGET DemandLoading PROPERTY SOURCES "${_sources}")
    target_sources(DemandLoading PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/CompletedTextureUpload.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/PreparedTextureUpload.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/BatchedTextureUpload.cpp")
    target_include_directories(DemandLoading PRIVATE "${_source_dir}/src" "${_source_dir}/src/Memory" "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_SOURCE_DIR}/zeno/include" "${CMAKE_SOURCE_DIR}/projects/rtxmg")
    # The optional numeric recorder uses the renderer's C++17 filesystem code.
    target_compile_features(DemandLoading PRIVATE cxx_std_17)
endfunction()

function(_zeno_otk_write_changed path contents)
    set(_previous "")
    if(EXISTS "${path}")
        file(READ "${path}" _previous)
    endif()
    if(NOT _previous STREQUAL contents)
        file(WRITE "${path}" "${contents}")
    endif()
endfunction()

function(zeno_otk_sync_events_compat)
    get_target_property(_source_dir DemandLoading SOURCE_DIR)
    set(_overlay "${CMAKE_CURRENT_BINARY_DIR}/otk_sync")
    set(_helper "${CMAKE_CURRENT_SOURCE_DIR}/TextureSyncEvent.h")
    # Quote-includes in SDK .cpp/.h files resolve beside the including file.
    # Mirror the private source tree so every TU sees the same PagingSystem
    # definition, including TUs already generated by the other adapters.
    file(GLOB_RECURSE _files CONFIGURE_DEPENDS RELATIVE "${_source_dir}/src"
        "${_source_dir}/src/*.cpp" "${_source_dir}/src/*.h")
    foreach(_relative IN LISTS _files)
        set(_original "${_source_dir}/src/${_relative}")
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_original}")
        file(READ "${_original}" _text)
        if(_relative STREQUAL "PagingSystem.h")
            set(_old "cuEventCreate( &event, CU_EVENT_DEFAULT )")
            string(FIND "${_text}" "${_old}" _found)
            if(_found EQUAL -1)
                message(FATAL_ERROR "OTK paging fence changed; review synchronization event flags")
            endif()
            string(REPLACE "${_old}" "cuEventCreate( &event, zeno::cluster::textureSyncEventFlags() )" _text "${_text}")
            string(PREPEND _text "#include \"${_helper}\"\n")
        elseif(_relative STREQUAL "ThreadPoolRequestProcessor.cpp")
            set(_old "cuCtxSetCurrent( context )")
            string(FIND "${_text}" "${_old}" _found)
            if(_found EQUAL -1)
                message(FATAL_ERROR "OTK request worker context changed; review context guard")
            endif()
            string(REPLACE "${_old}" "zeno::cluster::setTextureContext( context )" _text "${_text}")
            string(PREPEND _text "#include \"${_helper}\"\n")
        elseif(_relative STREQUAL "Util/ContextSaver.h")
            set(_old "cuCtxSetCurrent( m_context )")
            string(FIND "${_text}" "${_old}" _found)
            if(_found EQUAL -1)
                message(FATAL_ERROR "OTK saved context changed; review context guard")
            endif()
            string(REPLACE "${_old}" "zeno::cluster::setTextureContext( m_context )" _text "${_text}")
            string(PREPEND _text "#include \"${_helper}\"\n")
        endif()
        _zeno_otk_write_changed("${_overlay}/src/${_relative}" "${_text}")
    endforeach()
    get_target_property(_sources DemandLoading SOURCES)
    set(_adapted)
    foreach(_source IN LISTS _sources)
        if(IS_ABSOLUTE "${_source}")
            set(_absolute "${_source}")
        else()
            set(_absolute "${_source_dir}/${_source}")
        endif()
        file(RELATIVE_PATH _relative "${_source_dir}/src" "${_absolute}")
        if(NOT _relative MATCHES "^\\.\\./" AND _relative MATCHES "\\.cpp$" AND EXISTS "${_absolute}")
            list(APPEND _adapted "${_overlay}/src/${_relative}")
        else()
            list(APPEND _adapted "${_source}")
        endif()
    endforeach()
    set_property(TARGET DemandLoading PROPERTY SOURCES "${_adapted}")
    target_include_directories(DemandLoading BEFORE PRIVATE "${_overlay}/src" "${_overlay}/src/Memory")

    # MemoryPool is a public template: use one generated header for all its
    # consumers, rather than creating inconsistent inline definitions.
    get_target_property(_memory_dir Memory SOURCE_DIR)
    set(_original "${_memory_dir}/include/OptiXToolkit/Memory/MemoryPool.h")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_original}")
    file(READ "${_original}" _text)
    set(_old "cuEventCreate( &event, CU_EVENT_DEFAULT )")
    string(FIND "${_text}" "${_old}" _found)
    if(_found EQUAL -1)
        message(FATAL_ERROR "OTK memory fence changed; review synchronization event flags")
    endif()
    string(REPLACE "${_old}" "cuEventCreate( &event, zeno::cluster::textureSyncEventFlags() )" _text "${_text}")
    # Pool pressure can make an IO/upload worker wait for staged GPU blocks.
    # Keep that completion fence without holding a blocking driver wait.
    set(_old_wait "cuEventSynchronize( m_stagedBlocks.front().event );")
    string(FIND "${_text}" "${_old_wait}" _found)
    if(_found EQUAL -1)
        message(FATAL_ERROR "OTK memory wait changed; review background completion policy")
    endif()
    string(REPLACE "${_old_wait}" "OTK_ERROR_CHECK( zeno::cluster::waitBackgroundEvent( m_stagedBlocks.front().event ) );" _text "${_text}")
    string(PREPEND _text "#include \"${_helper}\"\n#include \"${CMAKE_SOURCE_DIR}/projects/rtxmg/BackgroundEventWait.h\"\n")
    _zeno_otk_write_changed("${_overlay}/include/OptiXToolkit/Memory/MemoryPool.h" "${_text}")
    target_include_directories(Memory BEFORE INTERFACE "$<BUILD_INTERFACE:${_overlay}/include>")
endfunction()

function(zeno_otk_nvrtc_headers output)
    set(_include "${CMAKE_SOURCE_DIR}/projects/optix-toolkit/DemandLoading/DemandLoading/include/")
    set(_generated "${CMAKE_CURRENT_BINARY_DIR}/otk_nvrtc/")
    foreach(_name Paging.h TextureSampler.h Texture2D.h ImageHelpers.h)
        if(_name STREQUAL "ImageHelpers.h")
            set(_include "${CMAKE_SOURCE_DIR}/projects/optix-toolkit/DemandLoading/ImageSource/include/")
            set(_relative "OptiXToolkit/ImageSource/${_name}")
        else()
            set(_relative "OptiXToolkit/DemandLoading/${_name}")
        endif()
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_include}${_relative}")
        file(READ "${_include}${_relative}" _text)
        if(_name STREQUAL "Paging.h")
            string(REPLACE "#include <cuda.h>" "#ifndef __CUDACC_RTC__\n#include <cuda.h>\n#endif" _text "${_text}")
        elseif(_name STREQUAL "TextureSampler.h")
            string(REPLACE "#include \"TextureDescriptor.h\"" "#ifndef __CUDACC_RTC__\n#include \"TextureDescriptor.h\"\n#endif" _text "${_text}")
        elseif(_name STREQUAL "Texture2D.h")
            # A null reference vector provides a read-only residency probe for
            # coarse fallbacks, without requesting all intermediate mip levels.
            string(REPLACE "pagingRequest( referenceBits," "if( referenceBits ) pagingRequest( referenceBits," _text "${_text}")
            # Trilinear samples need BOTH levels, including the mip-tail edge.
            string(REPLACE "return checkBitSet( sampler.startPage, residenceBits ) || isResident;" "return checkBitSet( sampler.startPage, residenceBits ) && isResident;" _text "${_text}")
            string(REPLACE "exp2f( fineLevel * 2.0f )" "exp2f( float(fineLevel) )" _text "${_text}")
            # Bilinear padding is measured in texels of the sampled mip, not
            # base-level texels. Cover the coarser trilinear level as well.
            string(REPLACE "0.5f * fmaxf( fabs( dPdx_x ), fabs( dPdy_x ) ) + ( 0.5f / sampler.width )" "0.5f * ( fabs( dPdx_x ) + fabs( dPdy_x ) ) + ( exp2f(float(fineLevel)) / sampler.width )" _text "${_text}")
            string(REPLACE "0.5f * fmaxf( fabs( dPdx_y ), fabs( dPdy_y ) ) + ( 0.5f / sampler.height )" "0.5f * ( fabs( dPdx_y ) + fabs( dPdy_y ) ) + ( exp2f(float(fineLevel)) / sampler.height )" _text "${_text}")
            string(REPLACE "isResident = requestTexFootprint2DRect( sampler, referenceBits, residenceBits, x, y, dxmax, dymax, fineLevel - 1, true );" "isResident = requestTexFootprint2DRect( sampler, referenceBits, residenceBits, x, y, dxmax, dymax, fineLevel - 1, true ) && isResident;" _text "${_text}")
            string(REPLACE "isResident = requestTexFootprint2DRect( sampler, referenceBits, residenceBits, x, y, dxmax, dymax, fineLevel + 2, true );" "isResident = requestTexFootprint2DRect( sampler, referenceBits, residenceBits, x, y, dxmax, dymax, fineLevel + 2, true ) && isResident;" _text "${_text}")
            # Both publication and hardware residency are required. A tile can
            # be physically mapped while its asynchronous copy is unfinished.
            # Retain the hardware LOD-boundary check and neighboring requests.
            set(_old_grad [=[    // If requestIfResident is false, use the predicated texture fetch to try and avoid requesting the footprint
    *isResident = !sampler->desc.isSparseTexture;
    if( !context.requestIfResident )
        rval = ::tex2DGrad<Sample>( sampler->texture, x, y, ddx, ddy, isResident );

    // Request the footprint if we don't know that it is resident (or if requestIfResident is true)
    if( *isResident == false && sampler->desc.isSparseTexture )
        *isResident = requestTexFootprint2DGrad( *sampler, context.referenceBits, context.residenceBits, x, y, ddx.x, ddx.y, ddy.x, ddy.y, 0.0f, 0.0f );

    // We know the footprint is resident, but we have not yet fetched the texture, so do it now.
    if( *isResident && context.requestIfResident )
        rval = ::tex2DGrad<Sample>( sampler->texture, x, y, ddx, ddy ); // non-pedicated texture fetch]=])
            set(_new_grad [=[    // Only published, completed uploads are eligible for sampling.
    const bool mapped = !sampler->desc.isSparseTexture || requestTexFootprint2DGrad( *sampler, context.referenceBits, context.residenceBits, x, y, ddx.x, ddx.y, ddy.x, ddy.y, 0.0f, 0.0f );
    *isResident = false;
    if( mapped )
        rval = ::tex2DGrad<Sample>( sampler->texture, x, y, ddx, ddy, isResident );
    if( sampler->desc.isSparseTexture && mapped && !*isResident )
    {
            const float ml = getMipLevel( ddx, ddy, sampler->width, sampler->height, 1.0f / sampler->desc.maxAnisotropy );
            const unsigned int first = isfinite(ml) ? static_cast<unsigned int>( fmaxf( floorf(ml) - 1.0f, 0.0f ) ) : sampler->mipTailFirstLevel;
            const float dx = fabsf(ddx.x) + fabsf(ddy.x) + exp2f(float(first+2)) / sampler->width;
            const float dy = fabsf(ddx.y) + fabsf(ddy.y) + exp2f(float(first+2)) / sampler->height;
            requestTexFootprint2DRect( *sampler, context.referenceBits, context.residenceBits, x, y, dx, dy, first, false );
            requestTexFootprint2DRect( *sampler, context.referenceBits, context.residenceBits, x, y, dx, dy, first + 2, false );
    }]=])
            string(FIND "${_text}" "${_old_grad}" _found)
            if(_found EQUAL -1)
                message(FATAL_ERROR "OTK gradient sampler changed; review hardware residency checks")
            endif()
            string(REPLACE "${_old_grad}" "${_new_grad}" _text "${_text}")
            string(REPLACE "*isResident = requestTexFootprint2DLod( *sampler, context.referenceBits, context.residenceBits, x, y, lod );" "requestTexFootprint2DLod( *sampler, context.referenceBits, context.residenceBits, x, y, lod );" _text "${_text}")
        else()
            string(REPLACE "INLINE CUarray_format_enum getFormat( float4" "#ifndef __CUDACC_RTC__\nINLINE CUarray_format_enum getFormat( float4" _text "${_text}")
            string(REPLACE "INLINE unsigned char toUChar( float value)" "#endif\nINLINE unsigned char toUChar( float value)" _text "${_text}")
        endif()
        set(_previous "")
        if(EXISTS "${_generated}${_relative}")
            file(READ "${_generated}${_relative}" _previous)
        endif()
        if(NOT _previous STREQUAL _text)
            file(WRITE "${_generated}${_relative}" "${_text}")
        endif()
    endforeach()
    set(${output} "${_generated}" PARENT_SCOPE)
endfunction()
