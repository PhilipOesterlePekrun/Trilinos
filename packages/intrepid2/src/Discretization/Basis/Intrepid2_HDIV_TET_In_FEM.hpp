// @HEADER
// *****************************************************************************
//                           Intrepid2 Package
//
// Copyright 2007 NTESS and the Intrepid2 contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

/** \file   Intrepid2_HDIV_TET_In_FEM.hpp
    \brief  Header file for the Intrepid2::Basis_HDIV_TET_In_FEM class.
    \author Created by R. Kirby and P. Bochev and D. Ridzal.
    Kokkorized by Kyungjoo Kim
 */

#ifndef __INTREPID2_HDIV_TET_IN_FEM_HPP__
#define __INTREPID2_HDIV_TET_IN_FEM_HPP__

#include "Intrepid2_Basis.hpp"
#include "Intrepid2_HGRAD_TET_Cn_FEM_ORTH.hpp"
#include "Intrepid2_HVOL_TRI_Cn_FEM.hpp"

#include "Intrepid2_PointTools.hpp"
#include "Teuchos_LAPACK.hpp"

namespace Intrepid2 {

/** \class  Intrepid2::Basis_HDIV_TET_In_FEM
      \brief  Implementation of the default H(div)-compatible Raviart-Thomas basis of arbitrary degree on Tetrahedral cells 

      Implements nodal basis of degree n (n>=1) on the reference Tetrahedron cell. The basis has
      cardinality n(n+1)(n+3)/2 and spans an INCOMPLETE polynomial
      space of degree n. Basis functions are dual to a
      unisolvent set of degrees-of-freedom (DoF) defined and
      enumerated as follows:

      \li The normal component on a lattice of order n+1 and
      offset 1 on each edge (see PointTools). This gives one point per edge in
      the lowest-order case.  These are the first
      4 * (n*(n+1)/2) degrees of freedom

      \li If n > 1, the x, y and z components at a lattice of
      order n+2 on the interior of the tetrahedron. These are the rest
      of the degrees of freedom.

      If the pointType argument to the constructor specifies equispaced points, 
      the the face and interior points will be equispaced.  If
      the pointType argument specifies warp-blend points, the
      interior of a warp-blend lattice will be used on each face
      and also for the cell interior.

 */

#define CardinalityHDivTet(order) (order*(order+1)*(order+3)/2)

namespace Impl {

/**
      \brief See Intrepid2::Basis_HDIV_TET_In_FEM
 */
class Basis_HDIV_TET_In_FEM {
public:

  /**
        \brief See Intrepid2::Basis_HDIV_TET_In_FEM
   */
  template<EOperator opType>
  struct Serial {
    template<typename outputValueViewType,
    typename inputPointViewType,
    typename workViewType,
    typename vinvViewType>
    KOKKOS_INLINE_FUNCTION
    static void
    getValues(       outputValueViewType outputValues,
        const inputPointViewType  inputPoints,
              workViewType        work,
        const vinvViewType        vinv );


    KOKKOS_INLINE_FUNCTION
    static ordinal_type
    getWorkSizePerPoint(ordinal_type order) {
      auto cardinality = CardinalityHDivTet(order);
      switch (opType) {
      case OPERATOR_DIV:
      case OPERATOR_D1:
        return 7*cardinality;
      default:
        return getDkCardinality<opType,3>()*cardinality;
      }
    }
  };

  template<typename DeviceType, ordinal_type numPtsPerEval,
  typename outputValueValueType, class ...outputValueProperties,
  typename inputPointValueType,  class ...inputPointProperties,
  typename vinvValueType,        class ...vinvProperties>
  static void
  getValues(       Kokkos::DynRankView<outputValueValueType,outputValueProperties...> outputValues,
      const Kokkos::DynRankView<inputPointValueType, inputPointProperties...>  inputPoints,
      const Kokkos::DynRankView<vinvValueType,       vinvProperties...>        vinv,
      const EOperator operatorType);

  /**
        \brief See Intrepid2::Basis_HDIV_TET_In_FEM
   */
  template<typename outputValueViewType,
  typename inputPointViewType,
  typename vinvViewType,
  typename workViewType,
  EOperator opType,
  ordinal_type numPtsEval>
  struct Functor {
    outputValueViewType _outputValues;
    const inputPointViewType  _inputPoints;
    const vinvViewType        _coeffs;
    workViewType        _work;

    KOKKOS_INLINE_FUNCTION
    Functor( outputValueViewType outputValues_,
        inputPointViewType  inputPoints_,
        vinvViewType        coeffs_,
        workViewType        work_)
    : _outputValues(outputValues_), _inputPoints(inputPoints_),
      _coeffs(coeffs_), _work(work_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const size_type iter) const {
      const auto ptBegin = Util<ordinal_type>::min(iter*numPtsEval,    _inputPoints.extent(0));
      const auto ptEnd   = Util<ordinal_type>::min(ptBegin+numPtsEval, _inputPoints.extent(0));

      const auto ptRange = Kokkos::pair<ordinal_type,ordinal_type>(ptBegin, ptEnd);
      const auto input   = Kokkos::subview( _inputPoints, ptRange, Kokkos::ALL() );

      typename workViewType::pointer_type ptr = _work.data() + _work.extent(0)*ptBegin*get_dimension_scalar(_work);

      auto vcprop = Kokkos::common_view_alloc_prop(_work);
      workViewType  work(Kokkos::view_wrap(ptr,vcprop), (ptEnd-ptBegin)*_work.extent(0));

      switch (opType) {
      case OPERATOR_VALUE : {
        auto output = Kokkos::subview( _outputValues, Kokkos::ALL(), ptRange, Kokkos::ALL() );
        Serial<opType>::getValues( output, input, work, _coeffs );
        break;
      }
      case OPERATOR_DIV: {
        auto output = Kokkos::subview( _outputValues, Kokkos::ALL(), ptRange);
        Serial<opType>::getValues( output, input, work, _coeffs );
        break;
      }
      default: {
        INTREPID2_TEST_FOR_ABORT( true,
            ">>> ERROR: (Intrepid2::Basis_HDIV_TET_In_FEM::Functor) operator is not supported");

      }
      }
    }
  };
};
}

template<typename DeviceType = void,
    typename outputValueType = double,
    typename pointValueType = double>
class Basis_HDIV_TET_In_FEM
    : public Basis<DeviceType,outputValueType,pointValueType> {
  public:
    using BasisBase = Basis<DeviceType, outputValueType, pointValueType>;
    using OrdinalTypeArray1DHost = typename BasisBase::OrdinalTypeArray1DHost;
    using OrdinalTypeArray2DHost = typename BasisBase::OrdinalTypeArray2DHost;
    using OrdinalTypeArray3DHost = typename BasisBase::OrdinalTypeArray3DHost;

    /** \brief  Constructor.
     */
    Basis_HDIV_TET_In_FEM(const ordinal_type order,
        const EPointType   pointType = POINTTYPE_EQUISPACED);

    using OutputViewType = typename BasisBase::OutputViewType;
    using PointViewType  = typename BasisBase::PointViewType;
    using ScalarViewType = typename BasisBase::ScalarViewType;
    using scalarType = typename BasisBase::scalarType;
    using BasisBase::getValues;

    virtual
    void
    getValues( /* */ OutputViewType outputValues,
        const PointViewType  inputPoints,
        const EOperator operatorType = OPERATOR_VALUE) const override {
#ifdef HAVE_INTREPID2_DEBUG
   Intrepid2::getValues_HDIV_Args(outputValues,
        inputPoints,
        operatorType,
        this->getBaseCellTopology(),
        this->getCardinality() );
#endif
    constexpr ordinal_type numPtsPerEval = Parameters::MaxNumPtsPerBasisEval;
    Impl::Basis_HDIV_TET_In_FEM::
    getValues<DeviceType,numPtsPerEval>( outputValues,
        inputPoints,
        this->coeffs_,
        operatorType);
     }

    virtual void 
    getScratchSpaceSize(      ordinal_type& perTeamSpaceSize,
                              ordinal_type& perThreadSpaceSize,
                        const PointViewType inputPointsconst,
                        const EOperator operatorType = OPERATOR_VALUE) const override;

    KOKKOS_INLINE_FUNCTION
    virtual void 
    getValues(       
          OutputViewType outputValues,
      const PointViewType  inputPoints,
      const EOperator operatorType,
      const typename Kokkos::TeamPolicy<typename DeviceType::execution_space>::member_type& team_member,
      const typename DeviceType::execution_space::scratch_memory_space & scratchStorage, 
      const ordinal_type subcellDim = -1,
      const ordinal_type subcellOrdinal = -1) const override;

    virtual
    void
    getDofCoords( ScalarViewType dofCoords ) const override {
#ifdef HAVE_INTREPID2_DEBUG
      // Verify rank of output array.
      INTREPID2_TEST_FOR_EXCEPTION( dofCoords.rank() != 2, std::invalid_argument,
          ">>> ERROR: (Intrepid2::Basis_HDIV_TET_In_FEM::getDofCoords) rank = 2 required for dofCoords array");
      // Verify 0th dimension of output array.
      INTREPID2_TEST_FOR_EXCEPTION( static_cast<ordinal_type>(dofCoords.extent(0)) != this->getCardinality(), std::invalid_argument,
          ">>> ERROR: (Intrepid2::Basis_HDIV_TET_In_FEM::getDofCoords) mismatch in number of dof and 0th dimension of dofCoords array");
      // Verify 1st dimension of output array.
      INTREPID2_TEST_FOR_EXCEPTION( dofCoords.extent(1) != this->getBaseCellTopology().getDimension(), std::invalid_argument,
          ">>> ERROR: (Intrepid2::Basis_HDIV_TET_In_FEM::getDofCoords) incorrect reference cell (1st) dimension in dofCoords array");
#endif
      Kokkos::deep_copy(dofCoords, this->dofCoords_);
    }

    virtual
    void
    getDofCoeffs( ScalarViewType dofCoeffs ) const override {
  #ifdef HAVE_INTREPID2_DEBUG
      // Verify rank of output array.
      INTREPID2_TEST_FOR_EXCEPTION( dofCoeffs.rank() != 2, std::invalid_argument,
          ">>> ERROR: (Intrepid2::Basis_HDIV_TET_In_FEM::getDofCoeffs) rank = 2 required for dofCoeffs array");
      // Verify 0th dimension of output array.
      INTREPID2_TEST_FOR_EXCEPTION( static_cast<ordinal_type>(dofCoeffs.extent(0)) != this->getCardinality(), std::invalid_argument,
        ">>> ERROR: (Intrepid2::Basis_HDIV_TET_In_FEM::getDofCoeffs) mismatch in number of dof and 0th dimension of dofCoeffs array");
      // Verify 1st dimension of output array.
      INTREPID2_TEST_FOR_EXCEPTION( dofCoeffs.extent(1) != this->getBaseCellTopology().getDimension(), std::invalid_argument,
        ">>> ERROR: (Intrepid2::Basis_HDIV_TET_In_FEM::getDofCoeffs) incorrect reference cell (1st) dimension in dofCoeffs array");
#endif
      Kokkos::deep_copy(dofCoeffs, this->dofCoeffs_);
    }

    void
    getExpansionCoeffs( ScalarViewType coeffs ) const {
      // has to be same rank and dimensions
      Kokkos::deep_copy(coeffs, this->coeffs_);
    }

    virtual
    const char*
    getName() const override {
      return "Intrepid2_HDIV_TET_In_FEM";
    }

    virtual
    bool
    requireOrientation() const override {
      return true;
    }

    /** \brief returns the basis associated to a subCell.

        The bases of the subCell are the restriction to the subCell of the bases of the parent cell,
        projected along normal to the subCell.

        \param [in] subCellDim - dimension of subCell
        \param [in] subCellOrd - position of the subCell among of the subCells having the same dimension
        \return pointer to the subCell basis of dimension subCellDim and position subCellOrd
     */
    BasisPtr<DeviceType,outputValueType,pointValueType>
      getSubCellRefBasis(const ordinal_type subCellDim, const ordinal_type subCellOrd) const override{

     if(subCellDim == 2) {
        return Teuchos::rcp(new
            Basis_HVOL_TRI_Cn_FEM<DeviceType,outputValueType,pointValueType>
            (this->basisDegree_-1, pointType_));
      }
      INTREPID2_TEST_FOR_EXCEPTION(true,std::invalid_argument,"Input parameters out of bounds");
    }

    BasisPtr<typename Kokkos::HostSpace::device_type,outputValueType,pointValueType>
    getHostBasis() const override{
      return Teuchos::rcp(new Basis_HDIV_TET_In_FEM<typename Kokkos::HostSpace::device_type,outputValueType,pointValueType>(this->basisDegree_, pointType_));
    }
  private:

    /** \brief expansion coefficients of the nodal basis in terms of the orthgonal one */
    Kokkos::DynRankView<scalarType,DeviceType> coeffs_;
  
    /** \brief type of lattice used for creating the DoF coordinates  */
    EPointType pointType_;
  };

}// namespace Intrepid2

#include "Intrepid2_HDIV_TET_In_FEMDef.hpp"

#endif
