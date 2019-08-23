using System;
using System.Collections;
using System.Collections.Generic;
using TensorFlow;
using Xunit;


namespace TensorFlowSharp.Tests.CSharp
{
    public class VariableTests
    {
        [Fact]
        public void ShouldNotShareVariablesSameTypeNoName()
        {
            using (var graph = new TFGraph())
            {
                var v1 = graph.Variable(graph.Const(0.5f));
                var v2 = graph.Variable(graph.Const(0.6f));

                using (var session = new TFSession(graph))
                {
                    session.GetRunner().AddTarget(graph.GetGlobalVariablesInitializer()).Run();
                    var result = session.GetRunner().Fetch(v1.Read, v2.Read).Run();
                    Assert.NotEqual(result[0].GetValue(), result[1].GetValue());
                }
            }
        }
        [Fact]
        public void ShouldNotShareVariablesSameType()
        {
            using (var graph = new TFGraph())
            {
                var v1 = graph.Variable(graph.Const(0.5f), operName: "v1");
                var v2 = graph.Variable(graph.Const(0.6f), operName: "v2");

                using (var session = new TFSession(graph))
                {
                    session.GetRunner().AddTarget(graph.GetGlobalVariablesInitializer()).Run();
                    var result = session.GetRunner().Fetch(v1.Read, v2.Read).Run();
                    Assert.NotEqual(result[0].GetValue(), result[1].GetValue());
                }
            }
        }

        [Fact]
        public void ShouldNotShareVariablesDifferentTypeNoName()
        {
            using (var graph = new TFGraph())
            {
                var v1 = graph.Variable(graph.Const(0.5f));
                var v2 = graph.Variable(graph.Const(0L));

                using (var session = new TFSession(graph))
                {
                    session.GetRunner().AddTarget(graph.GetGlobalVariablesInitializer()).Run();
                    var result = session.GetRunner().Fetch(v1.Read, v2.Read).Run();
                    Assert.NotEqual(result[0].TensorType, result[1].TensorType);
                }
            }
        }
        [Fact]
        public void ShouldNotShareVariablesDifferentType()
        {
            using (var graph = new TFGraph())
            {
                var v1 = graph.Variable(graph.Const(0.5f), operName: "v1");
                var v2 = graph.Variable(graph.Const(0L), operName: "v2");

                using (var session = new TFSession(graph))
                {
                    session.GetRunner().AddTarget(graph.GetGlobalVariablesInitializer()).Run();
                    var result = session.GetRunner().Fetch(v1.Read, v2.Read).Run();
                    Assert.NotEqual(result[0].TensorType, result[1].TensorType);
                }
            }
        }
    }
}
