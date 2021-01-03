

function Description() {
  return (
    <div className="App">
        <h1 class="heading" > Family Description </h1>
        <p class="p" >
            This page provides some family descriptions to make Protein Annotation accessible to non-experts.
        </p>
        
        <ol class="ol-class">
            <li class="family-accession" > Family Accession: PF01967.21</li>
            <p class="description" > Members of this family are involved in molybdenum biosynthesis. </p>
            
            <li class="family-accession" > Family Accession: PF13649.6</li>
            <p class="description" > Members of this family control the increase or decrease of production of some gene products. </p>
            
            <li class="family-accession" > Family Accession: PF03587.14</li>
            <p class="description" > This family helps Ribosomes to produce proteins.</p>
            
            <li class="family-accession" > Family Accession: PF16026.5</li>
            <p class="description" > The role of this family is to regulate mitochondrial quality.</p>
            
            <li class="family-accession" > Family Accession: PF02445.16</li>
            <p class="description" > This family accelerates some biosynthetic processes.</p>
        </ol>
    </div>
  );
}

export default Description;