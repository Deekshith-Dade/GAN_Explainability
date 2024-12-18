

networkParams_v1 =dict(
in_channels=[(32,32),
			 (32,32),
			 (64,64),
			 (64,64),
			 (128,128),
			 (128,128),
			 (256,256),
			 (256,256)],
out_channels=[(32,32),
			  (32,64),
			  (64,64),
			  (64,128),
			  (128,128),
			  (128,256),
			  (256,256),
			  (256,256)],
kernel_size =[3,
			  3,
			  3,
			  3,
			  3,
			  3,
			  3,
			  3],
padding	    =[1,
			  1,
			  1,
			  1,
			  1,
			  1,
			  1,
			  1],
numLayers = 8,
bias = False,
dropout = [True]*8
)

networkParams_v2 =dict(
in_channels=[(32,32),
			 (32,32),
			 (64,64),
			 (64,64),
			 (128,128),
			 (128,128),
			 (256,256),
			 (256,256),
			 (512,512),
			 (512,512)],
out_channels=[(32,32),
			  (32,64),
			  (64,64),
			  (64,128),
			  (128,128),
			  (128,256),
			  (256,256),
			  (256,512),
			  (512,512),
			  (512,512)],
kernel_size =[3,
			  3,
			  3,
			  3,
			  3,
			  3,
			  3,
			  3,
			  3,
			  3],
padding	    =[1,
			  1,
			  1,
			  1,
			  1,
			  1,
			  1,
			  1,
			  1,
			  1],
numLayers = 10,
bias = False,
dropout = [True]*10
)


networkParams_v3 =dict(
in_channels=[(32,32),
			 (32,32),
			 (32,32),
			 (32,32),
			 (64,64),
			 (64,64),
			 (64,64),
			 (64,64),
			 (128,128),
			 (128,128),
			 (128,128),
			 (128,128),
			 (256,256),
			 (256,256),
			 (256,256),
			 (512,512),
			 (512,512),
			 (512,512)],
out_channels=[(32,32),
			  (32,32),
			  (32,32),
			  (32,64),
			  (64,64),
			  (64,64),
			  (64,64),
			  (64,128),
			  (128,128),
			  (128,128),
			  (128,128),
			  (128,256),
			  (256,256),
			  (256,256),
			  (256,512),
			  (512,512),
			  (512,512),
			  (512,512)],
kernel_size =[3]*18,
padding	    =[1]*18,
numLayers = 18,
bias = False,
dropout = [True]*18
)

spatioTemporalParams_v1 =dict(
temporalResidualBlockParams = dict(
	in_channels=[(32,32),
				 (32,32),
				 (64,64),
				 (64,64),
				 (128,128),
				 (128,128),
				 (256,256),
				 (256,256)],
	out_channels=[(32,32),
				  (32,64),
				  (64,64),
				  (64,128),
				  (128,128),
				  (128,256),
				  (256,256),
				  (256,256)],
	kernel_size =[3,
				  3,
				  3,
				  3,
				  3,
				  3,
				  3,
				  3],
	padding	    =[1,
				  1,
				  1,
				  1,
				  1,
				  1,
				  1,
				  1],
	numLayers = 8,
	bias = False,
	dropout = [True]*8
	),	
spatialResidualBlockParams = dict(
	in_channels=[(32,32),
				 (32,32),
				 (64,64),
				 (64,64),
				 (128,128),
				 (128,128),
				 (256,256),
				 (256,256)],
	out_channels=[(32,32),
				  (32,64),
				  (64,64),
				  (64,128),
				  (128,128),
				  (128,256),
				  (256,256),
				  (256,256)],
	kernel_size =[7,
				  7,
				  7,
				  7,
				  7,
				  7,
				  7,
				  7],
	padding	    =[3,
				  3,
				  3,
				  3,
				  3,
				  3,
				  3,
				  3],
	numLayers = 8,
	bias = False,
	dropout = [True]*8
	)	
)


spatioTemporalParams_v3 =dict(
temporalResidualBlockParams = dict(
	in_channels=[(32,32),
				 (32,32),
				 (32,32),
				 (64,64),
				 (64,64),
				 (64,64),
				 (128,128),
				 (128,128),
				 (128,128),
				 (256,256),
				 (256,256),
				 (256,256)],
	out_channels=[(32,32),
				  (32,32),
				  (32,64),
				  (64,64),
				  (64,64),
				  (64,128),
				  (128,128),
				  (128,128),
				  (128,256),
				  (256,256),
				  (256,256),
				  (256,256)],
	kernel_size =[3]*12,
	padding	    =[1]*12,
	numLayers = 12,
	bias = False,
	dropout = [True]*12
	),	
spatialResidualBlockParams = dict(
	in_channels=[(32,32),
				 (32,32),
				 (32,32),
				 (64,64),
				 (64,64),
				 (64,64),
				 (128,128),
				 (128,128),
				 (128,128),
				 (256,256),
				 (256,256),
				 (256,256)],
	out_channels=[(32,32),
				  (32,32),
				  (32,64),
				  (64,64),
				  (64,64),
				  (64,128),
				  (128,128),
				  (128,128),
				  (128,256),
				  (256,256),
				  (256,256),
				  (256,256)],
	kernel_size =[7]*12,
	padding	    =[3]*12,
	numLayers = 12,
	bias = False,
	dropout = [True]*12
	)	
)


spatioTemporalParams_v4 =dict(
temporalResidualBlockParams = dict(
	blockType='Temporal',
	in_channels=[(32,32),
				 (64,64),
				 (128,128),
				 (256,256)],
	out_channels=[(32,64),
				  (64,128),
				  (128,256),
				  (256,256)],
	kernel_size =[3]*4,
	padding	    =[1]*4,
	numLayers = 4,
	bias = False,
	dropout = [True]*4
	),	
spatialResidualBlockParams = dict(
	blockType='Spatial',
	in_channels=[(32,32),
				 (64,64),
				 (128,128),
				 (256,256)],
	out_channels=[(32,64),
				  (64,128),
				  (128,256),
				  (256,256)],
	kernel_size =[7]*4,
	padding	    =[3]*4,
	numLayers = 4,
	bias = False,
	dropout = [True]*4
	)	
)

temporalTemporalParams_v1 =dict(
temporalResidualBlockParams = dict(
	blockType='Temporal',
	in_channels=[(32,32),
				 (64,64),
				 (128,128),
				 (256,256)],
	out_channels=[(32,64),
				  (64,128),
				  (128,256),
				  (256,256)],
	kernel_size =[3]*4,
	padding	    =[1]*4,
	numLayers = 4,
	bias = False,
	dropout = [True]*4
	),	
spatialResidualBlockParams = dict(
	blockType='Temporal',
	in_channels=[(32,32),
				 (64,64),
				 (128,128),
				 (256,256)],
	out_channels=[(32,64),
				  (64,128),
				  (128,256),
				  (256,256)],
	kernel_size =[7]*4,
	padding	    =[3]*4,
	numLayers = 4,
	bias = False,
	dropout = [True]*4
	)	
)