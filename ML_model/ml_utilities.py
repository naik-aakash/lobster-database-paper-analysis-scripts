#!/usr/bin/env python
# coding: utf-8

import warnings
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from skrebate import MultiSURFstar

warnings.filterwarnings("ignore")

shap.initjs()

logging.basicConfig(level=logging.INFO, format="%(message)s")

logging.info("Caution : Please change the number of parallel processes to run (n_jobs) in the ml_utilities.py module as per your system configuration. Default value is set to 30.")

n_jobs=30 #change the number of processes here

def grid_search(model, param, X_train, y_train):
    """
    Convenience function to setup inner cv pipeline for hyperparamter tuning
    """
    logging.info("Performing GridSearchCV")

    os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"

    # configure the cross-validation procedure
    np.random.seed(18012019)
    cv_inner = KFold(n_splits=5, shuffle=True, random_state=18012019)

    # setup model training pipeline
    pipeline = Pipeline([("selector", MultiSURFstar(n_jobs=-1)), ("regressor", model)])
    # define search
    search = GridSearchCV(
        pipeline,
        param_grid=param,
        scoring="neg_mean_absolute_error",
        cv=cv_inner,
        refit=True,
        return_train_score=True,
        n_jobs=n_jobs,
    )
    # execute search
    result = search.fit(X_train, y_train)

    logging.info("Finished GridSearchCV")
    # get the best performing model fit on the whole training set
    return result.best_estimator_, search


def get_feature_importance_plot(gridsearchcv_obj, modelname, features, iteration):
    """
    Function to save the plot top 20 features from randomforest algorithm with its scores
    """
    # get feature indices from train set included in model training
    selected_feature_indices = gridsearchcv_obj.best_estimator_.steps[0][
        1
    ].top_features_[: gridsearchcv_obj.best_estimator_.steps[0][1].n_features_to_select]

    # get feature importance score from randomforest model
    importances = gridsearchcv_obj.best_estimator_.steps[1][1].feature_importances_

    # Sort features indices based on their scores ()
    sorted_indices = importances.argsort()[::-1]
    top_indices = sorted_indices[:20]

    # get data in lists
    feature_score = importances[top_indices]
    feature_names = features.iloc[:, selected_feature_indices].columns[top_indices]

    # Create and save plot to neptune logger
    fig_feat = go.Figure(
        data=go.Bar(
            x=feature_score,
            y=feature_names,
            orientation="h",
        )
    )
    fig_feat.update_layout(yaxis=dict(tickfont=dict(size=18)))
    fig_feat.update_layout(xaxis=dict(tickfont=dict(size=18)))
    fig_feat.update_yaxes(title_font=dict(size=22), color="black")
    fig_feat.update_xaxes(title_font=dict(size=22), color="black")
    fig_feat.update_xaxes(
        showline=True, linewidth=1, linecolor="black", mirror=True, showgrid=False
    )
    fig_feat.update_yaxes(
        showline=True, linewidth=1, linecolor="black", mirror=True, showgrid=False
    )
    fig_feat.update_xaxes(ticks="inside", tickwidth=1, tickcolor="black", ticklen=5)
    fig_feat.update_yaxes(ticks="inside", tickwidth=1, tickcolor="black", ticklen=5)
    fig_feat.update_layout(template="simple_white")
    fig_feat.update_layout(width=1000, height=1000)
    fig_feat.update_layout(title_text="Feature scores", title_x=0.5)
    fig_feat.update_yaxes(autorange="reversed")
    fig_feat.write_html(
        "{}/{}_features_{}.html".format(modelname, modelname, iteration),
        include_mathjax="cdn",
    )
    fig_feat.write_image(
        "{}/{}_features_{}.svg".format(modelname, modelname, iteration),
        width=1000,
        height=1000,
    )

    logging.info("Saved RandomForestRegressor top 20 features plot")

    return fig_feat


def get_shap_plot(model, X_train, iteration, modelname):
    """
    Function to extract and store the shapley values plot to identify feature influence on model predictions
    """
    logging.info("Extracting Shapley values from the model")

    selected_feature_indices = np.argsort(
        -1 * model.best_estimator_.steps[0][1].feature_importances_
    )[: model.best_estimator_.steps[0][1].n_features_to_select]

    # Get the names of the selected features
    feature_names = [X_train.columns[i] for i in selected_feature_indices]

    # Extract shapley values from the best model

    explainer = shap.TreeExplainer(
        model.best_estimator_.steps[1][1], X_train.filter(feature_names)
    )
    shap_values = explainer.shap_values(
        X_train.filter(feature_names), check_additivity=False
    )

    fig = shap.summary_plot(
        shap_values,
        features=X_train.filter(feature_names),
        feature_names=X_train.filter(feature_names).columns,
        show=False,
    )
    plt.savefig("{}/{}_{}.svg".format(modelname, modelname, iteration))
    plt.close()

    logging.info("Done")

    return fig


def get_train_test_plot(test_errors, train_errors, labels, modelname):
    """
    Function to save the absolute error distribution violin plot
    """

    fig_val = go.Figure()

    fig_val.add_trace(
        go.Violin(
            x0=modelname,
            y=np.concatenate(test_errors),
            legendgroup="Test",
            name="Test",
            side="positive",
            hovertext=list(np.concatenate(labels)),
            line_color="blue",
            box_visible=True,
        )
    )
    fig_val.add_trace(
        go.Violin(
            x0=modelname,
            y=np.concatenate(train_errors),
            legendgroup="Train",
            name="Train",
            side="negative",
            line_color="orange",
            box_visible=True,
        )
    )
    fig_val.update_traces(meanline_visible=True)
    fig_val.update_layout(violingap=0, violinmode="overlay")
    fig_val.update_layout(
        xaxis_title="Model",
        yaxis_title="$\\text{Absolute errors }\\omega\\text{ }(cm^{⁻1})$",
    )
    fig_val.update_traces(marker_opacity=0.75)
    fig_val.update_layout(yaxis=dict(tickfont=dict(size=18)))
    fig_val.update_layout(xaxis=dict(tickfont=dict(size=18)))
    fig_val.update_yaxes(title_font=dict(size=22), color="black")
    fig_val.update_xaxes(title_font=dict(size=22), color="black")
    fig_val.update_layout(width=1000, height=1000)
    fig_val.update_xaxes(
        showline=True, linewidth=1, linecolor="black", mirror=True, showgrid=False
    )
    fig_val.update_yaxes(
        showline=True, linewidth=1, linecolor="black", mirror=True, showgrid=False
    )
    fig_val.update_xaxes(ticks="inside", tickwidth=1, tickcolor="black", ticklen=5)
    fig_val.update_yaxes(ticks="inside", tickwidth=1, tickcolor="black", ticklen=5)
    fig_val.update_layout(yaxis=dict(tickfont=dict(size=18)))
    fig_val.update_layout(yaxis=dict(tickfont=dict(size=18)))
    fig_val.update_layout(xaxis=dict(tickfont=dict(size=18)))
    fig_val.update_layout(template="simple_white")
    fig_val.update_layout(yaxis_zeroline=False)
    fig_val.write_html(
        "{}/{}_validation.html".format(modelname, modelname), include_mathjax="cdn"
    )
    fig_val.write_image(
        "{}/{}_validation.svg".format(modelname, modelname), width=1000, height=1000
    )

    logging.info("Saved model predictions absolute error distribution violin plot")

    return fig_val


def get_actual_predict_plot(df_predictions, modelname):
    """
    Function to save the actual and predicted values by the model for complete dataset
    """

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_predictions["lastphdospeak_actual"],
            y=df_predictions["lastphdospeak_predicted"],
            mode="markers",
            showlegend=False,
            hovertext=df_predictions.index
            + "<br>Composition :"
            + df_predictions.Composition,
        )
    )

    fig.update_traces(marker=dict(size=10, color="#1878b6"))

    lr = LinearRegression().fit(
        np.array(df_predictions["lastphdospeak_actual"]).reshape(-1, 1),
        df_predictions["lastphdospeak_predicted"],
    )
    y_hat = lr.predict(np.array(df_predictions["lastphdospeak_actual"]).reshape(-1, 1))

    r2 = round(
        lr.score(
            np.array(df_predictions["lastphdospeak_actual"]).reshape(-1, 1),
            df_predictions["lastphdospeak_predicted"],
        ),
        3,
    )

    fig.add_trace(
        go.Scatter(
            x=df_predictions["lastphdospeak_actual"],
            y=y_hat,
            mode="lines",
            showlegend=False,
            line_color="#f57f1f",
        )
    )

    fig.update_layout(
        xaxis_title="$\\text{Actual }\\omega\\text{ }(cm^{⁻1})$",
        yaxis_title="$\\text{Predicted }\\omega\\text{ } (cm^{⁻1})$",
    )
    fig.update_yaxes(title_font=dict(size=22), color="black")
    fig.update_xaxes(title_font=dict(size=22), color="black")
    fig.update_xaxes(showline=True, linewidth=1, linecolor="black", mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor="black", mirror=True)
    fig.update_xaxes(ticks="inside", tickwidth=1, tickcolor="black", ticklen=5)
    fig.update_yaxes(ticks="inside", tickwidth=1, tickcolor="black", ticklen=5)
    fig.update_layout(width=1000, height=1000)
    fig.update_layout(
        legend=dict(
            x=0,
            y=1,
            traceorder="normal",
            font=dict(family="sans-serif", size=20, color="black"),
        )
    )
    fig.add_annotation(
        xref="x domain",
        yref="y domain",
        x=0.90,
        y=0.5,
        text=r"$R^2={}$".format(r2),
        showarrow=False,
        font=dict(size=24, color="black"),
    )

    fig.update_layout(template="simple_white")
    fig.write_image(
        "{}/{}_predictions.pdf".format(modelname, modelname), width=1000, height=1000
    )
    fig.write_html(
        "{}/{}_predictions.html".format(modelname, modelname), include_mathjax="cdn"
    )

    df_predictions.to_csv("predictions_data.csv")

    logging.info(
        "Saved models predictions overview data as csv and scatter plot for overall dataset"
    )

    return fig


def get_metrics_df(abs_errors, rmse_scores, r2_scores, mape_scores, model):
    """
    Convenience function that evaluates cross validated model performance statistics and returns a pandas dataframe
    """
    # define the dataframe
    df = pd.DataFrame(index=[model])

    # map of metrics
    metrics = {
        "mae": abs_errors,
        "max_error": abs_errors,
        "rmse": rmse_scores,
        "r2": r2_scores,
        "mape": mape_scores,
    }

    def compute_stats(scores, metric_name):
        """
        Wrapper function to evaluate statistics using numpy methods
        """

        if metric_name == "mae":
            return {
                "mean": np.mean([np.mean(score) for score in scores]),
                "max": np.max([np.mean(score) for score in scores]),
                "min": np.min([np.mean(score) for score in scores]),
                "std": np.std([np.mean(score) for score in scores]),
            }

        if metric_name == "max_error":
            return {
                "mean": np.mean([np.max(score) for score in scores]),
                "max": np.max([np.max(score) for score in scores]),
                "min": np.min([np.max(score) for score in scores]),
                "std": np.std([np.max(score) for score in scores]),
            }

        else:
            return {
                "mean": np.mean(scores),
                "max": np.max(scores),
                "min": np.min(scores),
                "std": np.std(scores),
            }

    # loop to calculate and populate the metrics in the dataframe
    for metric, scores in metrics.items():
        for subset in ["train", "test"]:
            stats = compute_stats(scores=getattr(scores, subset), metric_name=metric)
            for stat_name, value in stats.items():
                df.loc[model, "{}_{}_{}".format(metric, subset, stat_name)] = value

    df.to_csv("summary_results.csv")

    logging.info("Computed model performance metrics stats and saved as a csv file")

    return df
