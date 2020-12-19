/* eslint-disable */
// this is an auto generated file. This will be overwritten

export const getProteinAnnotationTable = /* GraphQL */ `
  query GetProteinAnnotationTable($itemID: String!) {
    getProteinAnnotationTable(itemID: $itemID) {
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
export const listProteinAnnotationTables = /* GraphQL */ `
  query ListProteinAnnotationTables(
    $filter: TableProteinAnnotationTableFilterInput
    $limit: Int
    $nextToken: String
  ) {
    listProteinAnnotationTables(
      filter: $filter
      limit: $limit
      nextToken: $nextToken
    ) {
      items {
        itemID
        userID
        familyID
        familyAccession
        confidence
        description
        inputSequence
      }
      nextToken
    }
  }
`;
