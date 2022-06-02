///////////////////////////////////////////////////////////////////////////////
//               Dem Bones - Skinning Decomposition Library                  //
//         Copyright (c) 2019, Electronic Arts. All rights reserved.         //
///////////////////////////////////////////////////////////////////////////////



#ifndef DEM_BONES_INDEXING
#define DEM_BONES_INDEXING

#ifdef _MSC_VER
#pragma warning(disable:4172)
#endif

#include <Eigen/Dense>

namespace Dem
{
	
/** NullaryOp forward mapping for matrix with row indices and column indices,
	check: https://eigen.tuxfamily.org/dox/TopicCustomizing_NullaryExpr.html
*/
template<class ArgType, class RowIndexType, class ColIndexType>
class indexing_functor_row_col {
	const ArgType &m_arg;
	const RowIndexType &m_rowIndices;
	const ColIndexType &m_colIndices;
public:
	typedef Eigen::Matrix<typename ArgType::Scalar,
		RowIndexType::SizeAtCompileTime,
		ColIndexType::SizeAtCompileTime,
		ArgType::Flags& Eigen::RowMajorBit?Eigen::RowMajor:Eigen::ColMajor,
		RowIndexType::MaxSizeAtCompileTime,
		ColIndexType::MaxSizeAtCompileTime> MatrixType;
	indexing_functor_row_col(const ArgType& arg, const RowIndexType& row_indices, const ColIndexType& col_indices)
		: m_arg(arg), m_rowIndices(row_indices), m_colIndices(col_indices) {}
	const typename ArgType::Scalar& operator() (Eigen::Index row, Eigen::Index col) const {
		return m_arg(m_rowIndices[row], m_colIndices[col]);
	}
};

/**	Function forward mapping for matrix with row indices and column indices,
	check: https://eigen.tuxfamily.org/dox/TopicCustomizing_NullaryExpr.html
*/
template <class ArgType, class RowIndexType, class ColIndexType>
Eigen::CwiseNullaryOp<indexing_functor_row_col<ArgType, RowIndexType, ColIndexType>, typename indexing_functor_row_col<ArgType, RowIndexType, ColIndexType>::MatrixType>
indexing_row_col(const Eigen::MatrixBase<ArgType>& arg, const RowIndexType& row_indices, const ColIndexType& col_indices) {
	typedef indexing_functor_row_col<ArgType, RowIndexType, ColIndexType> Func;
	typedef typename Func::MatrixType MatrixType;
	return MatrixType::NullaryExpr(row_indices.size(), col_indices.size(), Func(arg.derived(), row_indices, col_indices));
}



/** NullaryOp forward mapping for matrix with row indices,
	check: https://eigen.tuxfamily.org/dox/TopicCustomizing_NullaryExpr.html
*/
template<class ArgType, class RowIndexType>
class indexing_functor_row {
	const ArgType &m_arg;
	const RowIndexType &m_rowIndices;
public:
	typedef Eigen::Matrix<typename ArgType::Scalar,
		RowIndexType::SizeAtCompileTime,
		ArgType::ColsAtCompileTime,
		ArgType::Flags& Eigen::RowMajorBit?Eigen::RowMajor:Eigen::ColMajor,
		RowIndexType::MaxSizeAtCompileTime,
		ArgType::MaxColsAtCompileTime> MatrixType;
	indexing_functor_row(const ArgType& arg, const RowIndexType& row_indices)
		: m_arg(arg), m_rowIndices(row_indices) {}
	const typename ArgType::Scalar& operator() (Eigen::Index row, Eigen::Index col) const {
		return m_arg(m_rowIndices[row], col);
	}
};

/** Function forward mapping for matrix with row indices,
	check: https://eigen.tuxfamily.org/dox/TopicCustomizing_NullaryExpr.html
*/
template <class ArgType, class RowIndexType>
Eigen::CwiseNullaryOp<indexing_functor_row<ArgType, RowIndexType>, typename indexing_functor_row<ArgType, RowIndexType>::MatrixType>
indexing_row(const Eigen::MatrixBase<ArgType>& arg, const RowIndexType& row_indices) {
	typedef indexing_functor_row<ArgType, RowIndexType> Func;
	typedef typename Func::MatrixType MatrixType;
	return MatrixType::NullaryExpr(row_indices.size(), arg.cols(), Func(arg.derived(), row_indices));
}



/** NullaryOp forward mapping for vector with indices,
	check: https://eigen.tuxfamily.org/dox/TopicCustomizing_NullaryExpr.html
*/
template<class ArgType, class IndexType>
class indexing_functor_vector {
	const ArgType &m_arg;
	const IndexType &m_indices;
public:
	typedef Eigen::Matrix<typename ArgType::Scalar,
		IndexType::SizeAtCompileTime,
		1,
		Eigen::ColMajor,
		IndexType::MaxSizeAtCompileTime, 
		1> VectorType;
	indexing_functor_vector(const ArgType& arg, const IndexType& indices)
		: m_arg(arg), m_indices(indices) {}
	const typename ArgType::Scalar& operator() (Eigen::Index idx) const {
		return m_arg(m_indices[idx]);
	}
};

/** Function forward mapping for vector with indices,
	check: https://eigen.tuxfamily.org/dox/TopicCustomizing_NullaryExpr.html
*/
template <class ArgType, class IndexType>
Eigen::CwiseNullaryOp<indexing_functor_vector<ArgType, IndexType>, typename indexing_functor_vector<ArgType, IndexType>::VectorType>
indexing_vector(const Eigen::MatrixBase<ArgType>& arg, const IndexType& indices) {
	typedef indexing_functor_vector<ArgType, IndexType> Func;
	typedef typename Func::VectorType VectorType;
	return VectorType::NullaryExpr(indices.size(), Func(arg.derived(), indices));
}

}

#endif
