/* eslint-disable */
// this is an auto generated file. This will be overwritten

export const onCreateProteinAnnotationTable = /* GraphQL */ `
  subscription OnCreateProteinAnnotationTable(
    $itemID: String
    $userID: String
    $familyID: String
    $familyAccession: String
    $confidence: String
  ) {
    onCreateProteinAnnotationTable(
      itemID: $itemID
      userID: $userID
      familyID: $familyID
      familyAccession: $familyAccession
      confidence: $confidence
    ) {
      itemID
      userID
      familyID
      familyAccession
      confidence
      description
      inputSequence
    }
  }
`;
export const onUpdateProteinAnnotationTable = /* GraphQL */ `
  subscription OnUpdateProteinAnnotationTable(
    $itemID: String
    $userID: String
    $familyID: String
    $familyAccession: String
    $confidence: String
  ) {
    onUpdateProteinAnnotationTable(
      itemID: $itemID
      userID: $userID
      familyID: $familyID
      familyAccession: $familyAccession
      confidence: $confidence
    ) {
      itemID
      userID
      familyID
      familyAccession
      confidence
      description
      inputSequence
    }
  }
`;
export const onDeleteProteinAnnotationTable = /* GraphQL */ `
  subscription OnDeleteProteinAnnotationTable(
    $itemID: String
    $userID: String
    $familyID: String
    $familyAccession: String
    $confidence: String
  ) {
    onDeleteProteinAnnotationTable(
      itemID: $itemID
      userID: $userID
      familyID: $familyID
      familyAccession: $familyAccession
      confidence: $confidence
    ) {
      itemID
      userID
      familyID
      familyAccession
      confidence
      description
      inputSequence
    }
  }
`;
