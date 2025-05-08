// @HEADER
// *****************************************************************************
//        MueLu: A package for multigrid based preconditioning
//
// Copyright 2012 NTESS and the MueLu contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_DefaultComm.hpp>

#include <MueLu_TestHelpers.hpp>
#include <MueLu_Version.hpp>

#include <Xpetra_MultiVectorFactory.hpp>
#include <Xpetra_VectorFactory.hpp>
#include <Xpetra_Vector.hpp>

#include <MueLu_UncoupledAggregationFactory.hpp>
#include <MueLu_CoalesceDropFactory.hpp>
#include <MueLu_AmalgamationFactory.hpp>
#include <MueLu_CoarseMapFactory.hpp>
#include <MueLu_Aggregates.hpp>
#include <MueLu_InterfaceAggregationFactory.hpp>

namespace MueLuTests {

TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(InterfaceAggregationFactory, Constructor, Scalar, LocalOrdinal, GlobalOrdinal, Node) {
#include <MueLu_UseShortNames.hpp>
  MUELU_TESTING_SET_OSTREAM;
  MUELU_TESTING_LIMIT_SCOPE(Scalar, GlobalOrdinal, Node);
  out << "version: " << MueLu::Version() << std::endl;

  RCP<InterfaceAggregationFactory> interfaceAggFact = rcp(new InterfaceAggregationFactory());
  TEST_EQUALITY(interfaceAggFact != Teuchos::null, true);
}

//
// 2) Test BuildBasedOnNodeMapping end-to-end on a simple 1D Poisson problem
//
TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(InterfaceAggregationFactory, BuildBasedOnNodeMapping, Scalar, LocalOrdinal, GlobalOrdinal, Node) {
  #include <MueLu_UseShortNames.hpp>
  MUELU_TESTING_SET_OSTREAM;
  MUELU_TESTING_LIMIT_SCOPE(Scalar, GlobalOrdinal, Node);

  Level level;
  const GlobalOrdinal nnodePrimal = 40;
  const GlobalOrdinal ndofnPrimal = 2;
  const GlobalOrdinal nnodeDual = 20;
  const GlobalOrdinal ndofnDual = 2;
  const GlobalOrdinal indexBase =  0;
  RCP<Matrix> A = TestHelpers::TestFactory<SC, LO, GO, NO>::Build1DPoisson(nnodePrimal*ndofnPrimal);
  A->SetFixedBlockSize(ndofnPrimal);
  level.Set("A", A);

  // --- b) Build a trivial primal-aggregation via UncoupledAggregationFactory ---
  RCP<AmalgamationFactory>    amalg    = rcp(new AmalgamationFactory());
  RCP<CoalesceDropFactory>    drop     = rcp(new CoalesceDropFactory());
  drop   ->SetFactory("UnAmalgamationInfo", amalg);
  RCP<UncoupledAggregationFactory> aggF = rcp(new UncoupledAggregationFactory());
  aggF   ->SetFactory("Graph",       drop);
  aggF   ->SetFactory("DofsPerNode", drop);
  aggF   ->SetMinNodesPerAggregate(1);
  aggF   ->SetMaxNeighAlreadySelected(0);
  aggF   ->SetOrdering("natural");
  
  //
  RCP<MultiVector> nullSpace = MultiVectorFactory::Build(A->getRowMap(), NSdim);
  nullSpace->randomize();
  fineLevel.Set("Nullspace", nullSpace);

  RCP<AmalgamationFactory> amalgFact = rcp(new AmalgamationFactory());
  RCP<CoalesceDropFactory> dropFact  = rcp(new CoalesceDropFactory());
  dropFact->SetFactory("UnAmalgamationInfo", amalgFact);

  RCP<UncoupledAggregationFactory> UncoupledAggFact = rcp(new UncoupledAggregationFactory());
  UncoupledAggFact->SetFactory("Graph", dropFact);
  UncoupledAggFact->SetFactory("DofsPerNode", dropFact);

  UncoupledAggFact->SetMinNodesPerAggregate(3);
  UncoupledAggFact->SetMaxNeighAlreadySelected(0);
  UncoupledAggFact->SetOrdering("natural");

  RCP<CoarseMapFactory> coarseMapFact = rcp(new CoarseMapFactory());
  coarseMapFact->SetFactory("Aggregates", UncoupledAggFact);

  RCP<BlockedCoarseMapFactory> blockedCoarseMapFact = rcp(new BlockedCoarseMapFactory());
  blockedCoarseMapFact->SetFactory("Aggregates", UncoupledAggFact);
  blockedCoarseMapFact->SetFactory("CoarseMap", coarseMapFact);
  //

  level.Request("Aggregates", aggF.get());
  aggF->Build(level);
  RCP<Aggregates> primalAgg = level.Get<RCP<Aggregates>>("Aggregates", aggF.get());

  // --- c) Supply an identity dual‐to‐primal node map on level 0 ---
  using Dual2Primal = std::map<LocalOrdinal, LocalOrdinal>;
  RCP<Dual2Primal> dual2primal = rcp(new Dual2Primal());
  LocalOrdinal localNumNodes = A->getRowMap()->getLocalNumElements();
  for (LocalOrdinal i = 0; i < localNumNodes; ++i) {
    (*dual2primal)[i] = i; // trivial identity
  }
  level.Set("DualNodeID2PrimalNodeID", dual2primal);

  // --- d) Configure and run your InterfaceAggregationFactory ---
  RCP<InterfaceAggregationFactory> interfaceAggFact = rcp(new InterfaceAggregationFactory());
  // parameters as in XML:
  interfaceAggFact->GetParameterList()->set("Dual/primal mapping strategy", "node-based");
  interfaceAggFact->GetParameterList()->set("number of DOFs per dual node", Teuchos::as<LocalOrdinal>(3));

  interfaceAggFact->SetFactory("A",                       NoFactory::get());
  interfaceAggFact->SetFactory("Aggregates",              aggF);
  interfaceAggFact->SetFactory("DualNodeID2PrimalNodeID", NoFactory::get());

  // request its outputs
  level.Request("Aggregates",     interfaceAggFact.get());
  level.Request("UnAmalgamationInfo", interfaceAggFact.get());
  interfaceAggFact->Build(level);

  // --- e) Verify that the dual‐side aggregates appeared and have the expected size ---
  RCP<Aggregates> dualAgg = level.Get<RCP<Aggregates>>("Aggregates", interfaceAggFact.get());
  // Here we expect exactly the same number of aggregates as on the primal side.
  TEST_EQUALITY(dualAgg->GetNumAggregates(), primalAgg->GetNumAggregates());
}

#define MUELU_ETI_GROUP(Scalar, LocalOrdinal, GlobalOrdinal, Node)                                                      \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(InterfaceAggregationFactory, Constructor, Scalar, LocalOrdinal, GlobalOrdinal, Node) \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(InterfaceAggregationFactory, BuildBasedOnNodeMapping, Scalar, LocalOrdinal, GlobalOrdinal, Node)

#include <MueLu_ETI_4arg.hpp>

}  // namespace MueLuTests