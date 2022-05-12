// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package kernels

import (
	"github.com/apache/arrow/go/v9/arrow/compute"
	"github.com/apache/arrow/go/v9/arrow/compute/exec/functions"
	"github.com/apache/arrow/go/v9/arrow/scalar"
)

func trivialScalarUnaryAsArrayExec(kernelExec functions.ArrayKernelExec, hndl compute.NullHandling) functions.ArrayKernelExec {
	return func(ctx *compute.KernelCtx, batch *compute.ExecBatch, out compute.Datum) error {
		if out.Kind() == compute.KindArray {
			return kernelExec(ctx, batch, out)
		}

		if hndl == compute.NullIntersection && !batch.Values[0].(*compute.ScalarDatum).Value.IsValid() {
			out.(*compute.ScalarDatum).Value.SetValid(false)
			return nil
		}

		arrInput, err := scalar.MakeArrayFromScalar(batch.Values[0].(*compute.ScalarDatum).Value, 1, ctx.Ctx.Mem)
		if err != nil {
			return err
		}
		defer arrInput.Release()

		arrOutput, err := scalar.MakeArrayFromScalar(out.(*compute.ScalarDatum).Value, 1, ctx.Ctx.Mem)
		if err != nil {
			return err
		}
		defer arrOutput.Release()

		inDat := compute.NewDatum(arrInput)
		defer inDat.Release()
		outDat := compute.NewDatum(arrOutput)
		defer outDat.Release()
		err = kernelExec(ctx, &compute.ExecBatch{Values: []compute.Datum{inDat}, Length: 1}, outDat)
		if err != nil {
			return err
		}

		val, err := scalar.GetScalar(arrOutput, 0)
		if err != nil {
			return err
		}

		prev := out.(*compute.ScalarDatum).Value
		out.(*compute.ScalarDatum).Value = val
		if rel, ok := prev.(scalar.Releasable); ok {
			rel.Release()
		}
		return nil
	}
}

func firstType(_ *compute.KernelCtx, descrs []compute.ValueDescr) (compute.ValueDescr, error) {
	result := descrs[0]
	result.Shape = compute.GetBroadcastShape(descrs)
	return result, nil
}
