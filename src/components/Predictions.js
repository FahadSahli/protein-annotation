import { useState, useEffect } from 'react';
import { API, graphqlOperation  } from 'aws-amplify';
import { listProteinAnnotationTables } from './graphql/queries';
import './styles.css';

//const initialState  = { name: '', description: '' }
//var showTable = false;

function Predictions(props) {
  
  //const [formState, setFormState] = useState(initialState)
  const [predictions, setPredictions] = useState([]);
  const currentUserID = props["userID"];
  
  
  useEffect(() => {
    fetchPredictions();
    //setTimeout(window.location.reload(), 60000);
    //window.setTimeout(function(){window.location.reload(false)}, 3000);
  }, []);

  async function fetchPredictions() {
    try {
      const todoData = await API.graphql(graphqlOperation(listProteinAnnotationTables));
      const predictions = todoData.data.listProteinAnnotationTables.items;
      //showTable = true;
      
      console.log("Predictions...", predictions);
      console.log("lenfth...", predictions.length);
      console.log("userID...",  currentUserID);
      
      setPredictions(predictions);
    } catch (err) { console.log('error fetching predictions') }
  }

  return (
    <div className="App">
      <div style={{marginBottom: 30}}>
        <h1>Predictions</h1>
        <table class="table">
          <tr>
            <th>Input Sequence</th>
            <th>Family Accession</th>
            <th>Confidence</th>
          </tr>
          {
            predictions.map((elm, i) => {
              return currentUserID === elm.userID ?
                <tr>
                  <td class="limit-length">{elm.inputSequence}</td>
                  <td>{elm.familyAccession}</td>
                  <td>{elm.confidence}</td>
                </tr>
                :
                <nobr></nobr>;
                }
              )
            }
        </table>
      </div> 
    </div>
  );
}

export default Predictions;