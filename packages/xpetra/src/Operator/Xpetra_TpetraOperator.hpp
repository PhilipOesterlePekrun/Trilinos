// @HEADER
// *****************************************************************************
//             Xpetra: A linear algebra interface package
//
// Copyright 2012 NTESS and the Xpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef XPETRA_TPETRAOPERATOR_HPP
#define XPETRA_TPETRAOPERATOR_HPP

#include "Xpetra_TpetraConfigDefs.hpp"

#include <Tpetra_Operator.hpp>
#include <Tpetra_Details_residual.hpp>

#include "Xpetra_Map.hpp"
#include "Xpetra_TpetraMap.hpp"
#include "Xpetra_MultiVector.hpp"
#include "Xpetra_TpetraMultiVector.hpp"
#include "Xpetra_Operator.hpp"

#include "Xpetra_Utils.hpp"

namespace Xpetra {

template <class Scalar,
          class LocalOrdinal,
          class GlobalOrdinal,
          class Node = Tpetra::KokkosClassic::DefaultNode::DefaultNodeType>
class TpetraOperator : public Operator<Scalar, LocalOrdinal, GlobalOrdinal, Node> {
 public:
  //@{

  //! The Map associated with the domain of this operator, which must be compatible with X.getMap().
  virtual const Teuchos::RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> > getDomainMap() const {
    XPETRA_MONITOR("TpetraOperator::getDomainMap()");
    return toXpetra(op_->getDomainMap());
  }

  //! The Map associated with the range of this operator, which must be compatible with Y.getMap().
  virtual const Teuchos::RCP<const Map<LocalOrdinal, GlobalOrdinal, Node> > getRangeMap() const {
    XPETRA_MONITOR("TpetraOperator::getRangeMap()");
    return toXpetra(op_->getRangeMap());
  }

  //! \brief Computes the operator-multivector application.
  /*! Loosely, performs \f$Y = \alpha \cdot A^{\textrm{mode}} \cdot X + \beta \cdot Y\f$. However, the details of operation
      vary according to the values of \c alpha and \c beta. Specifically
      - if <tt>beta == 0</tt>, apply() <b>must</b> overwrite \c Y, so that any values in \c Y (including NaNs) are ignored.
      - if <tt>alpha == 0</tt>, apply() <b>may</b> short-circuit the operator, so that any values in \c X (including NaNs) are ignored.
   */
  virtual void
  apply(const Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> &X,
        Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> &Y,
        Teuchos::ETransp mode = Teuchos::NO_TRANS,
        Scalar alpha          = Teuchos::ScalarTraits<Scalar>::one(),
        Scalar beta           = Teuchos::ScalarTraits<Scalar>::zero()) const {
    op_->apply(toTpetra(X), toTpetra(Y), mode, alpha, beta);
  }

  /// \brief Whether this operator supports applying the transpose or conjugate transpose.
  virtual bool hasTransposeApply() const {
    return op_->hasTransposeApply();
  }

  //@}

  //! @name Overridden from Teuchos::Describable
  //@{

  //! A simple one-line description of this object.
  std::string description() const {
    XPETRA_MONITOR("TpetraOperator::description");
    return op_->description();
  }

  //! Print the object with the given verbosity level to a FancyOStream.
  void describe(Teuchos::FancyOStream &out, const Teuchos::EVerbosityLevel verbLevel = Teuchos::Describable::verbLevel_default) const {
    XPETRA_MONITOR("TpetraOperator::describe");
    op_->describe(out, verbLevel);
  }

  //@}

  //! @name Xpetra specific
  //@{

  TpetraOperator() {}

  //! TpetraOperator constructor to wrap a Tpetra::Operator object
  TpetraOperator(const Teuchos::RCP<Tpetra::Operator<Scalar, LocalOrdinal, GlobalOrdinal, Node> > &op)
    : op_(op) {}  // TODO removed const

  //! Gets the operator out
  virtual RCP<Tpetra::Operator<Scalar, LocalOrdinal, GlobalOrdinal, Node> > getOperator() { return op_; }

  //! Gets the operator out
  virtual RCP<const Tpetra::Operator<Scalar, LocalOrdinal, GlobalOrdinal, Node> > getOperatorConst() const { return op_; }

  //! Compute a residual R = B - (*this) * X
  void residual(const Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> &X,
                const Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> &B,
                Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> &R) const {
    Tpetra::Details::residual(*op_, toTpetra(X), toTpetra(B), toTpetra(R));
  }

  //@}

 private:
  //! The Tpetra::Operator which this class wraps.
  RCP<Tpetra::Operator<Scalar, LocalOrdinal, GlobalOrdinal, Node> > op_;

};  // TpetraOperator class

#if ((!defined(HAVE_TPETRA_INST_SERIAL)) && (!defined(HAVE_TPETRA_INST_INT_INT)) && defined(HAVE_XPETRA_EPETRA))
// specialization for Tpetra Map on EpetraNode and GO=int
template <>
class TpetraOperator<double, int, int, EpetraNode>
  : public Operator<double, int, int, EpetraNode> {
 public:
  typedef double Scalar;
  typedef int GlobalOrdinal;
  typedef int LocalOrdinal;
  typedef EpetraNode Node;

  //@{

  //! The Map associated with the domain of this operator, which must be compatible with X.getMap().
  virtual const Teuchos::RCP<const Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node> > getDomainMap() const {
    return Teuchos::null;
  }

  //! The Map associated with the range of this operator, which must be compatible with Y.getMap().
  virtual const Teuchos::RCP<const Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node> > getRangeMap() const {
    return Teuchos::null;
  }

  //! \brief Computes the operator-multivector application.
  /*! Loosely, performs \f$Y = \alpha \cdot A^{\textrm{mode}} \cdot X + \beta \cdot Y\f$. However, the details of operation
      vary according to the values of \c alpha and \c beta. Specifically
      - if <tt>beta == 0</tt>, apply() <b>must</b> overwrite \c Y, so that any values in \c Y (including NaNs) are ignored.
      - if <tt>alpha == 0</tt>, apply() <b>may</b> short-circuit the operator, so that any values in \c X (including NaNs) are ignored.
   */
  virtual void
  apply(const Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> &X,
        Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> &Y,
        Teuchos::ETransp mode = Teuchos::NO_TRANS,
        Scalar alpha          = Teuchos::ScalarTraits<Scalar>::one(),
        Scalar beta           = Teuchos::ScalarTraits<Scalar>::zero()) const {}

  /// \brief Whether this operator supports applying the transpose or conjugate transpose.
  virtual bool hasTransposeApply() const { return false; }

  //@}

  //! @name Overridden from Teuchos::Describable
  //@{

  //! A simple one-line description of this object.
  std::string description() const { return std::string(""); }

  //! Print the object with the given verbosity level to a FancyOStream.
  void describe(Teuchos::FancyOStream &out, const Teuchos::EVerbosityLevel verbLevel = Teuchos::Describable::verbLevel_default) const {}

  //@}

  //! @name Xpetra specific
  //@{

  //! TpetraOperator constructor to wrap a Tpetra::Operator object
  TpetraOperator(const Teuchos::RCP<Tpetra::Operator<Scalar, LocalOrdinal, GlobalOrdinal, Node> > &op) {}

  //! Gets the operator out
  RCP<Tpetra::Operator<Scalar, LocalOrdinal, GlobalOrdinal, Node> > getOperator() { return Teuchos::null; }

  void residual(const Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> &X,
                const Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> &B,
                Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> &R) const {
  }

  //@}

};  // TpetraOperator class
#endif

#if ((!defined(HAVE_TPETRA_INST_SERIAL)) && (!defined(HAVE_TPETRA_INST_INT_LONG_LONG)) && defined(HAVE_XPETRA_EPETRA))
// specialization for Tpetra Map on EpetraNode and GO=int
template <>
class TpetraOperator<double, int, long long, EpetraNode>
  : public Operator<double, int, long long, EpetraNode> {
 public:
  typedef double Scalar;
  typedef long long GlobalOrdinal;
  typedef int LocalOrdinal;
  typedef EpetraNode Node;

  //@{

  //! The Map associated with the domain of this operator, which must be compatible with X.getMap().
  virtual const Teuchos::RCP<const Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node> > getDomainMap() const {
    return Teuchos::null;
  }

  //! The Map associated with the range of this operator, which must be compatible with Y.getMap().
  virtual const Teuchos::RCP<const Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node> > getRangeMap() const {
    return Teuchos::null;
  }

  //! \brief Computes the operator-multivector application.
  /*! Loosely, performs \f$Y = \alpha \cdot A^{\textrm{mode}} \cdot X + \beta \cdot Y\f$. However, the details of operation
      vary according to the values of \c alpha and \c beta. Specifically
      - if <tt>beta == 0</tt>, apply() <b>must</b> overwrite \c Y, so that any values in \c Y (including NaNs) are ignored.
      - if <tt>alpha == 0</tt>, apply() <b>may</b> short-circuit the operator, so that any values in \c X (including NaNs) are ignored.
   */
  virtual void
  apply(const Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> &X,
        Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> &Y,
        Teuchos::ETransp mode = Teuchos::NO_TRANS,
        Scalar alpha          = Teuchos::ScalarTraits<Scalar>::one(),
        Scalar beta           = Teuchos::ScalarTraits<Scalar>::zero()) const {}

  /// \brief Whether this operator supports applying the transpose or conjugate transpose.
  virtual bool hasTransposeApply() const { return false; }

  //@}

  //! @name Overridden from Teuchos::Describable
  //@{

  //! A simple one-line description of this object.
  std::string description() const { return std::string(""); }

  //! Print the object with the given verbosity level to a FancyOStream.
  void describe(Teuchos::FancyOStream &out, const Teuchos::EVerbosityLevel verbLevel = Teuchos::Describable::verbLevel_default) const {}

  //@}

  //! @name Xpetra specific
  //@{

  //! TpetraOperator constructor to wrap a Tpetra::Operator object
  TpetraOperator(const Teuchos::RCP<Tpetra::Operator<Scalar, LocalOrdinal, GlobalOrdinal, Node> > &op) {}

  //! Gets the operator out
  RCP<Tpetra::Operator<Scalar, LocalOrdinal, GlobalOrdinal, Node> > getOperator() { return Teuchos::null; }

  void residual(const Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> &X,
                const Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> &B,
                Xpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node> &R) const {
  }
  //@}

};  // TpetraOperator class
#endif

}  // namespace Xpetra

#define XPETRA_TPETRAOPERATOR_SHORT
#endif  // XPETRA_TPETRAOPERATOR_HPP
