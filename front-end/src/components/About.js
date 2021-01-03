

function About() {
  return (
    <div className="App">
        <h1 class="heading" > About </h1>
        <p class="p" >
            This web application helps molecular biologists annotate their protein sequences. Protein annotation is an important step in drug development and involves manual process. The manual process 
            is where molecular biologists compare a query protein sequence with other sequences, with known classes, to determine the most similar sequence to the query. Then, the query protein is 
            assigned the class of the similar sequence. The tool is developed to minimize this manual process. It is highly accurate with an accuracy score of more than 92% on the test set of Pfam data. 
            The following is a set of FAQs that would help you get more information about the tool.
        </p>
        <ol>
          <li class="question" > Q: Who are the users? </li>
          <p class="answer" > The users are molecular biology researchers and practitioners who work in drug design and development. </p>
          
          <li class="question" > Q: How can the tool handle my research and experiments? </li>
          <p class="answer" > The tool considers part of the research process which is protein annotation. It performs the annotation accurately and fast, so manual annotation is reduced. This helps our 
                users to invest more time and effort focusing on other parts of their research.</p>
          
          <li class="question" > Q: Why to use the tool? </li>
          <p class="answer" > The tool has an accuracy of higher than 92% which means that the manual process done by molecular biologists is minimal. Also, the tool is 64.6% faster than the available 
                tools, so our users do not encounter delays when they need to annotate their protein data. </p>
          
          <li class="question" > Q: How can I be confident about the results provided by the tool?</li>
          <p class="answer" > With every prediction, the tool provides a score between 0 and 1, inclusive. The closer the score to 1, the more confident the tool with the corresponding prediction. </p>
          
          <li class="question" > Q: Is there a learning curve for the tool?</li>
          <p class="answer" > No, there is no learning curve to use the tool. Users just access the <a href="https://master.d3a8o6ga50nbtf.amplifyapp.com/">web page</a>, upload their files, and then 
                they get their results. </p>
          
          <li class="question" > Q: How to use the tool? </li>
          <p class="answer" > Users can go to the <a href="https://master.d3a8o6ga50nbtf.amplifyapp.com/">web application</a> and upload their data, and the application will provide them with the 
                annotations. </p>
          
          <li class="question" > Q: How can I upload my data?</li>
          <p class="answer" > When you visit the site, you will be prompted to upload your files. </p>
          
          <li class="question" > Q: What are the formats of documents to be uploaded? </li>
          <p class="answer" > Currently, the tool only accepts Comma Separated Value (CSV) documents. </p>
        </ol>
    </div>
  );
}

export default About;