/*!
 * Copyright by Contributors
 * \file extension.h
 * \brief some extension of expressions,
 *  used to support something beyond elementwise op
 * \author Tianqi Chen, Bing Xu
 */
#ifndef MSHADOW_EXTENSION_H_
#define MSHADOW_EXTENSION_H_
#include "./expr_engine-inl.h"
#include "./extension/broadcast.h"
#include "./extension/unpack_patch2col.h"
#include "./extension/pack_col2patch.h"
#include "./extension/reshape.h"
#include "./extension/swapaxis.h"
#include "./extension/reduceto1d.h"
#include "./extension/spatial_pool.h"
#include "./extension/spatial_unpool.h"
#include "./extension/channel_pool.h"
#include "./extension/channel_unpool.h"
#include "./extension/pad.h"
#include "./extension/crop.h"
#include "./extension/mirror.h"
#include "./extension/concat.h"
#include "./extension/implicit_gemm.h"
#include "./extension/choose.h"
#include "./extension/one_hot.h"
#include "./extension/slice.h"
#endif  // MSHADOW_EXTENSION_H_
