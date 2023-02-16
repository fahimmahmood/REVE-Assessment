import torch

class RNN(torch.nn.Module):
    
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        '''
        Parameters:
            input_dim: No. of unique tokens in input vocabulary. 
            embedding_dim: size of embedding layer i.e. dimension of vector space. Words are mapped \
                vector spaces.
            hidden_dim: No. of units in hidden layer.  
            output_dim: No. of classes 
        '''
        super().__init__()
        # convert integer-encoded input tokens to vectors of fixed size. This is layer is trained
        # to learn the optimal representations of input tokens.
        self.embedding = torch.nn.Embedding(input_dim, embedding_dim)

        # Compute the hidden state from input sequence.
        self.rnn = torch.nn.LSTM(embedding_dim,
                                 hidden_dim)        
        # Maps final hidden state to the output layer
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        

    def forward(self, text):
        # text dim: [sentence length, batch size]
        
        embedded = self.embedding(text)
        # embedded dim: [sentence length, batch size, embedding dim]
        
        output, (hidden, cell) = self.rnn(embedded)
        # output dim: [sentence length, batch size, hidden dim]
        # hidden dim: [1, batch size, hidden dim]

        hidden.squeeze_(0)
        # hidden dim: [batch size, hidden dim]
        
        output = self.fc(hidden)
        return output

if __name__=="__main__":

    input_dim, embedding_dim, hidden_dim, output_dim, batch_size, seq_len = \
        100, 50, 32, 2, 4, 10

    dummy_input = torch.randint(
        low=0,
        high=input_dim,
        size=(seq_len,batch_size)
    )
    model = RNN(input_dim,embedding_dim,hidden_dim,output_dim)

    
    print("Input Shape: ", dummy_input.shape) #Input Shape:  torch.Size([10, 4])

    embedded = model.embedding(dummy_input)
    '''
    The shape of embedded input text after passing through embedding layer is
    [sentence_length,batch_size,embedding_dim].
    snetence_length = longest sentence in the batch
    batch_size = no. of sentence in the batch
    embedding_dim = dimension of word embedding
    '''
    print("Embedded Shape: ",embedded.shape) #Embedded Shape:  torch.Size([10, 4, 50])


    output, (hidden,cell) = model.rnn(embedded)
    '''
    output shape is [sentence_length,batch_size,hidden_dim]
    hidden_dim denotes the hidden units in LSTM layer where each element in output tensor 
    is the hidden state of LSTM cell.
    '''
    print("Output Shape: ", output.shape) #Output Shape:  torch.Size([10, 4, 32])

    '''
    Shape of final hidden state of LSTM after processing entire input sequence is 
    hidden.shape i.e. [timestep,batch_size,hidden_dim].
    '''
    print("Hidden Shape: ",hidden.shape) #Hidden Shape:  torch.Size([1, 4, 32])
    print("Cell Shape: ",cell.shape) #Cell Shape:  torch.Size([1, 4, 32])
    
    '''
    To ignore timestep and only get final hidden steps of LSTM, we squeeze the first 
    dimension of hidden tensor.
    '''
    hidden.squeeze_(0)
    print("Hidden Shape after squeeze: ",hidden.shape) #Hidden Shape after squeeze:  torch.Size([4, 32])

    '''
    Shape of output of fully connected layer in the form [batch_size,output_dim]
    '''
    output = model.fc(hidden)
    print("Output Shape after fc: ",output.shape) #Output Shape after fc:  torch.Size([4, 2])

